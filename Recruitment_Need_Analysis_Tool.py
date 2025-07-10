from __future__ import annotations


from streamlit import session_state as ss
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore

import asyncio
import json
import re
import ast
import logging
import os
from dataclasses import dataclass
from typing import Any, Literal, cast

from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
import httpx
import streamlit as st
from openai import AsyncOpenAI
import importlib.util
from dotenv import load_dotenv
from dateutil import parser as dateparser
import datetime as dt
import csv
import base64

_spec = importlib.util.spec_from_file_location(
    "file_tools", Path(__file__).with_name("file_tools.py")
)
assert _spec is not None
_file_tools = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_file_tools)
extract_text_from_file = _file_tools.extract_text_from_file

SCHEMA: dict[str, list[dict[str, str]]] = {}
KEY_TO_STEP: dict[str, str] = {}
with open("wizard_schema.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        step = row["step"]
        if step not in SCHEMA:
            SCHEMA[step] = []
        # Parsing options (as list if present)
        options = row["options"].split(";") if row["options"].strip() else None
        row["options"] = options
        SCHEMA[step].append(row)
        KEY_TO_STEP[row["key"]] = step

DATE_KEYS = {"date_of_employment_start", "application_deadline", "probation_period"}
INDUSTRY_OPTIONS = [
    "IT",
    "Finance",
    "Healthcare",
    "Manufacturing",
    "Retail",
]
DOMAIN_OPTIONS = [
    "AI",
    "Cloud",
    "Cybersecurity",
    "Data Engineering",
    "Analytics",
]

st.markdown(
    """
    <style>
    /* rotes Stern-Prefix erzeugt roten Rahmen, wenn das Feld leer ist */
    input.must_req:placeholder-shown {
        border: 1px solid #e74c3c !important;   /* Streamlit default überschreiben */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── OpenAI setup ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY fehlt! Bitte in .env oder secrets.toml eintragen.")
    st.stop()

client = AsyncOpenAI(api_key=api_key)


async def generate_text(
    prompt: str,
    *,
    model: str = "gpt-4o",
    temperature: float = 0.5,
    max_tokens: int = 800,
    system_msg: str = "You are a helpful assistant.",
) -> str:
    """Send a prompt to OpenAI and return the reply."""

    chat = await client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()


# ── JSON helpers ──────────────────────────────────────────────────────────────
def brute_force_brace_fix(s: str) -> str:
    opens, closes = s.count("{") - s.count("}"), s.count("[") - s.count("]")
    return s + ("}" * max(opens, 0)) + ("]" * max(closes, 0))


def safe_json_load(text: str) -> dict:
    """
    Scrub GPT output into valid JSON, or return {}.
    """
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        cleaned2 = re.sub(r",\s*([}\]])", r"\1", cleaned).replace("'", '"')
        try:
            return json.loads(cleaned2)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(cleaned2)
            except Exception:
                try:
                    return json.loads(brute_force_brace_fix(cleaned2))
                except Exception as e:
                    logging.error("Secondary JSON extraction failed: %s", e)
                    return {}


# ★ mandatory
MUST_HAVE_KEYS = {
    "job_title",
    "company_name",
    "city",
    "contract_type",
    "seniority_level",
    "role_description",
    "role_type",
    "task_list",
    "must_have_skills",
    "salary_range",
    "salary_currency",
    "pay_frequency",
    "recruitment_contact_email",
}

ORDER = [
    "BASIC",
    "COMPANY",
    "ROLE",
    "SKILLS",
    "BENEFITS",
    "INTERVIEW",
    "SUMMARY",
]

STEP_TITLES = {
    "COMPANY": "Company & Department",
    "ROLE": "Role & Tasks",
}

STEPS: list[tuple[str, list[str]]] = [
    (
        STEP_TITLES.get(name, name.title().replace("_", " ")),
        [item["key"] for item in SCHEMA[name]],
    )
    for name in ORDER
    if name in SCHEMA
]


def group_by_step(
    extracted: dict[str, "ExtractResult"],
) -> dict[str, dict[str, "ExtractResult"]]:
    """Group flat extraction results by wizard step."""

    grouped: dict[str, dict[str, ExtractResult]] = {
        step: {} for step in ORDER if step in SCHEMA
    }
    for key, res in extracted.items():
        step = KEY_TO_STEP.get(key)
        if step:
            grouped.setdefault(step, {})[key] = res
    return grouped


# ──────────────────────────────────────────────
# REGEX PATTERNS
# (complete list incl. addons for missing keys)
# ──────────────────────────────────────────────
# helper to cut boilerplate
def _simple(label_en: str, label_de: str, cap: str) -> str:
    return rf"(?:{label_en}|{label_de})\s*:?\s*(?P<{cap}>.+)"


REGEX_PATTERNS = {
    # BASIC INFO - mandatory
    "job_title": _simple("Job\\s*Title|Position|Stellenbezeichnung", "", "job_title"),
    "employment_type": _simple(
        "Employment\\s*Type",
        "Vertragsart|Beschäftigungsart|Arbeitszeit",
        "employment_type",
    ),
    "contract_type": _simple(
        "Contract\\s*Type", "Vertragstyp|Anstellungsart", "contract_type"
    ),
    "seniority_level": _simple(
        "Seniority\\s*Level", "Karrierelevel", "seniority_level"
    ),
    "date_of_employment_start": _simple(
        "Start\\s*Date|Begin\\s*Date", "Eintrittsdatum", "date_of_employment_start"
    ),
    "work_schedule": _simple("Work\\s*Schedule", "Arbeitszeitmodell", "work_schedule"),
    "work_location_city": _simple("City|Ort", "Ort", "work_location_city"),
    # Company core
    "company_name": _simple("Company|Employer", "Unternehmen", "company_name"),
    "city": _simple("City", "Stadt", "city"),
    "company_size": _simple("Company\\s*Size", "Mitarbeiterzahl", "company_size"),
    "industry": _simple("Industry", "Branche", "industry"),
    "headquarters_location": _simple(
        "HQ\\s*Location", "Hauptsitz", "headquarters_location"
    ),
    "place_of_work": _simple("Place\\s*of\\s*Work", "Arbeitsort", "place_of_work"),
    "company_website": r"(?P<company_website>https?://\S+)",
    # Department / team
    "department_name": _simple("Department", "Abteilung", "department_name"),
    "brand_name": _simple("Brand", "", "brand_name"),
    "team_size": _simple("Team\\s*Size", "Teamgröße", "team_size"),
    "team_structure": _simple("Team\\s*Structure", "Teamaufbau", "team_structure"),
    "direct_reports_count": _simple(
        "Direct\\s*Reports", "Direkt\\s*Berichte", "direct_reports_count"
    ),
    "reports_to": _simple("Reports\\s*To", "unterstellt", "reports_to"),
    "supervises": _simple("Supervises", "Führungsverantwortung", "supervises"),
    "tech_stack": _simple("Tech(ology)?\\s*Stack", "Technologien?", "tech_stack"),
    "culture_notes": _simple("Culture", "Kultur", "culture_notes"),
    "team_challenges": _simple(
        "Team\\s*Challenges", "Herausforderungen", "team_challenges"
    ),
    "client_difficulties": _simple(
        "Client\\s*Difficulties", "Kundenprobleme", "client_difficulties"
    ),
    "main_stakeholders": _simple(
        "Stakeholders?", "Hauptansprechpartner", "main_stakeholders"
    ),
    "team_motivation": _simple(
        "Team\\s*Motivation", "Team\\s*Motivationen?", "team_motivation"
    ),
    "recent_team_changes": _simple(
        "Recent\\s*Team\\s*Changes", "Teamveränderungen", "recent_team_changes"
    ),
    "office_language": _simple("Office\\s*Language", "Bürosprache", "office_language"),
    "office_type": _simple("Office\\s*Type", "Bürotyp", "office_type"),
    # Role definition
    "role_description": _simple(
        "Role\\s*Description|Role\\s*Purpose", "Aufgabenstellung", "role_description"
    ),
    "role_type": _simple("Role\\s*Type", "Rollenart", "role_type"),
    "role_keywords": _simple(
        "Role\\s*Keywords?", "Stellenschlüsselwörter", "role_keywords"
    ),
    "role_performance_metrics": _simple(
        "Performance\\s*Metrics", "Rollenkennzahlen", "role_performance_metrics"
    ),
    "role_priority_projects": _simple(
        "Priority\\s*Projects", "Prioritätsprojekte", "role_priority_projects"
    ),
    "primary_responsibilities": _simple(
        "Primary\\s*Responsibilities", "Hauptaufgaben", "primary_responsibilities"
    ),
    "key_deliverables": _simple(
        "Key\\s*Deliverables", "Ergebnisse", "key_deliverables"
    ),
    "success_metrics": _simple(
        "Success\\s*Metrics", "Erfolgskennzahlen", "success_metrics"
    ),
    "main_projects": _simple("Main\\s*Projects", "Hauptprojekte", "main_projects"),
    "travel_required": _simple(
        "Travel\\s*Required", "Reisetätigkeit", "travel_required"
    ),
    "physical_duties": _simple(
        "Physical\\s*Duties", "Körperliche\\s*Arbeit", "physical_duties"
    ),
    "on_call": _simple("On[-\\s]?Call", "Bereitschaft", "on_call"),
    "decision_authority": _simple(
        "Decision\\s*Authority", "Entscheidungsbefugnis", "decision_authority"
    ),
    "process_improvement": _simple(
        "Process\\s*Improvement", "Prozessverbesserung", "process_improvement"
    ),
    "innovation_expected": _simple(
        "Innovation\\s*Expected", "Innovationsgrad", "innovation_expected"
    ),
    "daily_tools": _simple("Daily\\s*Tools", "Tägliche\\s*Tools?", "daily_tools"),
    # Tasks
    "task_list": _simple("Task\\s*List", "Aufgabenliste", "task_list"),
    "key_responsibilities": _simple(
        "Key\\s*Responsibilities", "Hauptverantwortlichkeiten", "key_responsibilities"
    ),
    "technical_tasks": _simple(
        "Technical\\s*Tasks?", "Technische\\s*Aufgaben", "technical_tasks"
    ),
    "managerial_tasks": _simple(
        "Managerial\\s*Tasks?", "Führungsaufgaben", "managerial_tasks"
    ),
    "administrative_tasks": _simple(
        "Administrative\\s*Tasks?", "Verwaltungsaufgaben", "administrative_tasks"
    ),
    "customer_facing_tasks": _simple(
        "Customer[-\\s]?Facing\\s*Tasks?",
        "Kundenkontaktaufgaben",
        "customer_facing_tasks",
    ),
    "internal_reporting_tasks": _simple(
        "Internal\\s*Reporting\\s*Tasks", "Berichtsaufgaben", "internal_reporting_tasks"
    ),
    "performance_tasks": _simple(
        "Performance\\s*Tasks", "Leistungsaufgaben", "performance_tasks"
    ),
    "innovation_tasks": _simple(
        "Innovation\\s*Tasks", "Innovationsaufgaben", "innovation_tasks"
    ),
    "task_prioritization": _simple(
        "Task\\s*Prioritization", "Aufgabenpriorisierung", "task_prioritization"
    ),
    # Skills
    "must_have_skills": _simple(
        "Must[-\\s]?Have\\s*Skills?", "Erforderliche\\s*Kenntnisse", "must_have_skills"
    ),
    "nice_to_have_skills": _simple(
        "Nice[-\\s]?to[-\\s]?Have\\s*Skills?", "Wünschenswert", "nice_to_have_skills"
    ),
    "hard_skills": _simple("Hard\\s*Skills", "Fachkenntnisse", "hard_skills"),
    "soft_skills": _simple("Soft\\s*Skills", "Soziale\\s*Kompetenzen?", "soft_skills"),
    "certifications_required": _simple(
        "Certifications?\\s*Required", "Zertifikate", "certifications_required"
    ),
    "language_requirements": _simple(
        "Language\\s*Requirements", "Sprachanforderungen", "language_requirements"
    ),
    "languages_optional": _simple(
        "Languages\\s*Optional", "Weitere\\s*Sprachen", "languages_optional"
    ),
    "analytical_skills": _simple(
        "Analytical\\s*Skills", "Analytische\\s*Fähigkeiten", "analytical_skills"
    ),
    "communication_skills": _simple(
        "Communication\\s*Skills", "Kommunikationsfähigkeiten", "communication_skills"
    ),
    "project_management_skills": _simple(
        "Project\\s*Management\\s*Skills",
        "Projektmanagementskills?",
        "project_management_skills",
    ),
    "tool_proficiency": _simple(
        "Tool\\s*Proficiency", "Toolkenntnisse", "tool_proficiency"
    ),
    "domain_expertise": _simple(
        "Domain\\s*Expertise", "Fachgebiet", "domain_expertise"
    ),
    "leadership_competencies": _simple(
        "Leadership\\s*Competencies", "Führungskompetenzen?", "leadership_competencies"
    ),
    "industry_experience": _simple(
        "Industry\\s*Experience", "Branchenerfahrung", "industry_experience"
    ),
    "soft_requirement_details": _simple(
        "Soft\\s*Requirement\\s*Details",
        "Weitere\\s*Anforderungen",
        "soft_requirement_details",
    ),
    "years_experience_min": _simple(
        "Years\\s*Experience", "Berufserfahrung", "years_experience_min"
    ),
    "it_skills": _simple("IT\\s*Skills", "IT[-\\s]?Kenntnisse", "it_skills"),
    "visa_sponsorship": _simple(
        "Visa\\s*Sponsorship", "Visasponsoring", "visa_sponsorship"
    ),
    # Compensation
    "salary_currency": _simple("Currency", "Währung", "salary_currency"),
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|bis|to|–)\s*\d{4,6})",
    "salary_range_min": r"(?P<salary_range_min>\d{4,6})\s*(?:-|bis|to|–)\s*\d{4,6}",
    "salary_range_max": r"\d{4,6}\s*(?:-|bis|to|–)\s*(?P<salary_range_max>\d{4,6})",
    "bonus_scheme": _simple(
        "Bonus\\s*Scheme|Bonus\\s*Model", "Bonusregelung", "bonus_scheme"
    ),
    "commission_structure": _simple(
        "Commission\\s*Structure", "Provisionsmodell", "commission_structure"
    ),
    "variable_comp": _simple(
        "Variable\\s*Comp", "Variable\\s*Vergütung", "variable_comp"
    ),
    "vacation_days": _simple("Vacation\\s*Days", "Urlaubstage", "vacation_days"),
    "remote_policy": _simple(
        "Remote\\s*Policy", "Home\\s*Office\\s*Regelung", "remote_policy"
    ),
    "flexible_hours": _simple(
        "Flexible\\s*Hours|Gleitzeit", "Gleitzeit", "flexible_hours"
    ),
    "relocation_support": _simple(
        "Relocation\\s*Support", "Umzugshilfe", "relocation_support"
    ),
    "childcare_support": _simple(
        "Childcare\\s*Support", "Kinderbetreuung", "childcare_support"
    ),
    "learning_budget": _simple(
        "Learning\\s*Budget", "Weiterbildungsbudget", "learning_budget"
    ),
    "company_car": _simple("Company\\s*Car", "Firmenwagen", "company_car"),
    "sabbatical_option": _simple(
        "Sabbatical\\s*Option", "Auszeitmodell", "sabbatical_option"
    ),
    "health_insurance": _simple(
        "Health\\s*Insurance", "Krankenversicherung", "health_insurance"
    ),
    "pension_plan": _simple("Pension\\s*Plan", "Altersvorsorge", "pension_plan"),
    "stock_options": _simple("Stock\\s*Options", "Aktienoptionen", "stock_options"),
    "other_perks": _simple("Other\\s*Perks", "Weitere\\s*Benefits", "other_perks"),
    "pay_frequency": r"(?P<pay_frequency>monthly|annual|yearly|hourly|quarterly)",
    # Recruitment
    "recruitment_contact_email": r"(?P<recruitment_contact_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "recruitment_contact_phone": _simple(
        "Contact\\s*Phone", "Telefon", "recruitment_contact_phone"
    ),
    "recruitment_steps": _simple(
        "Recruitment\\s*Steps", "Bewerbungsprozess", "recruitment_steps"
    ),
    "recruitment_timeline": _simple(
        "Recruitment\\s*Timeline", "Bewerbungszeitplan", "recruitment_timeline"
    ),
    "number_of_interviews": _simple(
        "Number\\s*of\\s*Interviews", "Anzahl\\s*Interviews", "number_of_interviews"
    ),
    "interview_format": _simple(
        "Interview\\s*Format", "Interviewformat", "interview_format"
    ),
    "interview_stage_count": _simple(
        "Interview\\s*Stages?", "Bewerbungsgespräche", "interview_stage_count"
    ),
    "interview_docs_required": _simple(
        "Interview\\s*Docs\\s*Required", "Unterlagen", "interview_docs_required"
    ),
    "assessment_tests": _simple(
        "Assessment\\s*Tests?", "Einstellungstests?", "assessment_tests"
    ),
    "interview_notes": _simple(
        "Interview\\s*Notes", "Interviewnotizen", "interview_notes"
    ),
    "onboarding_process": _simple(
        "Onboarding\\s*Process", "Einarbeitung", "onboarding_process"
    ),
    "onboarding_process_overview": _simple(
        "Onboarding\\s*Overview",
        "Einarbeitungsüberblick",
        "onboarding_process_overview",
    ),
    "probation_period": _simple("Probation\\s*Period", "Probezeit", "probation_period"),
    "mentorship_program": _simple(
        "Mentorship\\s*Program", "Mentorenprogramm", "mentorship_program"
    ),
    "welcome_package": _simple(
        "Welcome\\s*Package", "Willkommenspaket", "welcome_package"
    ),
    "application_instructions": _simple(
        "Application\\s*Instructions", "Bewerbungshinweise", "application_instructions"
    ),
    # Key contacts
    "line_manager_name": _simple(
        "Line\\s*Manager", "Fachvorgesetzte?r", "line_manager_name"
    ),
    "line_manager_email": r"(?P<line_manager_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "hr_poc_name": _simple("HR\\s*POC", "Ansprechpartner\\s*HR", "hr_poc_name"),
    "hr_poc_email": r"(?P<hr_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "finance_poc_name": _simple(
        "Finance\\s*POC", "Ansprechpartner\\s*Finance", "finance_poc_name"
    ),
    "finance_poc_email": r"(?P<finance_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
}


LLM_PROMPT = (
    "You parse German or English job ads. "
    "Return ONLY valid JSON where every key maps to an object "
    'with fields "value" (string|null) and "confidence" (0-1).'
)

# Additional lightweight patterns without explicit labels
FALLBACK_PATTERNS: dict[str, str] = {
    "employment_type": r"(?P<employment_type>Vollzeit|Teilzeit|Werkstudent(?:[ei]n)?|Praktikum|Mini[-\s]?Job|Freelance|Internship|Full[-\s]?time|Part[-\s]?time)",
    "contract_type": r"(?P<contract_type>unbefristet|befristet|festanstellung|permanent|temporary|fixed[-\s]?term|contract|freelancer|project|werkvertrag|zeitarbeit)",
    "seniority_level": r"(?P<seniority_level>Junior|Mid|Senior|Lead|Head|Manager|Einsteiger|Berufserfahren)",
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|bis|to|–)\s*\d{4,6})",
    "work_location_city": r"\bin\s+(?P<work_location_city>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,})?)",
}


def search_company_name(text: str) -> ExtractResult | None:
    pat_bei = (
        r"(?<=bei\s)"
        r"(?P<company_name>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&., \-]{2,}\s*(?:GmbH|AG|KG|SE|Inc\.|Ltd\.|LLC|e\.V\.))"
        r"(?=\s|$)"
    )
    m = re.search(pat_bei, text)
    if m:
        return ExtractResult(m.group("company_name"), 0.8)

    pat_generic = (
        r"(?P<company_name>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&., \-]{2,}\s*(?:GmbH|AG|KG|SE|Inc\.|Ltd\.|LLC|e\.V\.))"
        r"(?=\s|$)"
    )
    m = re.search(pat_generic, text)
    if m:
        return ExtractResult(m.group("company_name"), 0.7)
    return None


def search_city(text: str) -> ExtractResult | None:
    """Return a city found after the word 'in'."""

    pat = (
        r"\bin\s+"
        r"(?P<work_location_city>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,}(?:\s+"
        r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,})?)"
    )
    m = re.search(pat, text)
    if m:
        return ExtractResult(m.group("work_location_city"), 0.7)
    return None


# ── Utility dataclass ─────────────────────────────────────────────────────────
@dataclass
class ExtractResult:
    value: str | None = None
    confidence: float = 0.0


# HTML-to-text helper


def html_text(html: str) -> str:
    """Return visible text only."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    return " ".join(soup.stripped_strings)


# ── Regex search --------------------------------------------------------------
def pattern_search(text: str, key: str, pat: str) -> ExtractResult:
    """Return a :class:`ExtractResult` for the first regex match.

    Common prefixes such as ``Name:`` or ``City:`` are stripped from the
    resulting value. A confidence of ``0.9`` is returned for regex matches.
    """
    m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if not (m and m.group(key)):
        return ExtractResult(None, 0.0)

    val = m.group(key).strip()

    # gängige Labels am Zeilenanfang entfernen
    val = re.sub(r"^(?:Name|City|Ort|Stadt)\s*[:\-]?\s*", "", val, flags=re.I)

    return ExtractResult(value=val, confidence=0.9)


# ── Cached loaders ------------------------------------------------------------
@st.cache_data(ttl=24 * 60 * 60)
def http_text(url: str) -> str:
    try:
        html = httpx.get(url, timeout=20).text
    except httpx.HTTPError as e:
        logging.error("HTTP request failed: %s", e)
        return ""
    return html_text(html)


@st.cache_data(ttl=24 * 60 * 60)
def pdf_text(data: BytesIO) -> str:
    """Cached wrapper around :func:`extract_text_from_file` for PDFs."""
    return extract_text_from_file(data.getvalue(), "application/pdf")


@st.cache_data(ttl=24 * 60 * 60)
def docx_text(data: BytesIO) -> str:
    """Cached wrapper around :func:`extract_text_from_file` for DOCX."""
    return extract_text_from_file(
        data.getvalue(),
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )


# ── Skill helpers -------------------------------------------------------------
def parse_skill_list(raw: str | list[str] | None) -> list[str]:
    """Return a cleaned list of skills from various input formats."""
    if raw is None:
        return []
    if isinstance(raw, list):
        items = raw
    else:
        items = re.split(r"[;,\n]+", str(raw))
    return [s.strip() for s in items if s and s.strip()]


def selectable_buttons(
    options: list[str], label: str, session_key: str, cols: int = 3
) -> list[str]:
    """Render options as toggleable buttons and store selections.

    Args:
        options: Suggestions to present.
        label: Section label shown above the buttons.
        session_key: Key used in :mod:`streamlit.session_state`.
        cols: Number of columns used for layout.

    Returns:
        The updated list of selected options.
    """

    st.write(label)
    selected = cast(list[str], ss.setdefault(session_key, []))
    columns = st.columns(cols)
    for idx, opt in enumerate(options):
        col = columns[idx % cols]
        is_selected = opt in selected
        btn_type = "primary" if is_selected else "secondary"
        btn_label = f"✓ {opt}" if is_selected else opt
        if col.button(
            btn_label,
            key=f"{session_key}_{idx}",
            type=cast(Literal["primary", "secondary", "tertiary"], btn_type),
            use_container_width=True,
        ):
            if is_selected:
                selected.remove(opt)
            else:
                selected.append(opt)

    ss[session_key] = selected
    return selected


async def _suggest_skills(data: dict, kind: str, count: int) -> list[str]:
    existing: list[str] = []
    for key in [
        "must_have_skills",
        "nice_to_have_skills",
        "hard_skills",
        "soft_skills",
    ]:
        existing.extend(parse_skill_list(data.get(key)))

    prompt = (
        f"List the top {count} {kind} skills for a job titled '{data.get('job_title', '')}'. "
        f"Consider role type '{data.get('role_type', '')}', role description '{data.get('role_description', '')}' "
        f"and tasks '{data.get('task_list', '')}'. Exclude: {', '.join(existing)}. "
        'Return JSON object {"skills": [..]} with one skill per list item.'
    )

    chat = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are an expert HR assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = safe_json_load(chat.choices[0].message.content or "")
    return [s.strip() for s in raw.get("skills", []) if s]


async def suggest_hard_skills(data: dict) -> list[str]:
    """Suggest up to 10 relevant hard skills."""
    return await _suggest_skills(data, "hard", 10)


async def suggest_soft_skills(data: dict) -> list[str]:
    """Suggest up to 5 relevant soft skills."""
    return await _suggest_skills(data, "soft", 5)


async def _suggest_benefits(data: dict, mode: str, count: int) -> list[str]:
    """Return a list of benefits based on the provided mode."""

    job_title = data.get("job_title", "")
    location = data.get("city") or data.get("work_location_city", "")

    if mode == "title":
        prefix = f"for the job title '{job_title}'"
    elif mode == "location":
        prefix = f"commonly offered by employers in {location}"
    else:
        prefix = f"competitors usually offer for similar '{job_title}' roles"

    prompt = (
        f"List up to {count} employee benefits {prefix}. "
        'Return JSON object {"benefits": [..]} with one benefit per list item.'
    )

    chat = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are an expert HR assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )

    raw = safe_json_load(chat.choices[0].message.content or "")
    return [b.strip() for b in raw.get("benefits", []) if b]


async def suggest_benefits_by_title(data: dict, count: int) -> list[str]:
    """Suggest benefits tailored to the job title."""

    return await _suggest_benefits(data, "title", count)


async def suggest_benefits_by_location(data: dict, count: int) -> list[str]:
    """Suggest benefits typical for the given location."""

    return await _suggest_benefits(data, "location", count)


async def suggest_benefits_competitors(data: dict, count: int) -> list[str]:
    """Suggest benefits that competitors offer for similar roles."""

    return await _suggest_benefits(data, "competitors", count)


# New team context suggestion helpers
async def _suggest_items(prompt: str, key: str) -> list[str]:
    chat = await client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "You are an expert HR assistant."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = safe_json_load(chat.choices[0].message.content or "")
    return [str(x).strip() for x in raw.get(key, []) if x]


async def suggest_team_challenges(data: dict, count: int = 5) -> list[str]:
    """Suggest common team challenges."""

    prompt = (
        f"List up to {count} challenges for a team in the {data.get('industry', '')} industry. "
        'Return JSON object {"items": [..]} with one challenge per list item.'
    )
    return await _suggest_items(prompt, "items")


async def suggest_client_difficulties(data: dict, count: int = 5) -> list[str]:
    """Suggest frequent client difficulties."""

    prompt = (
        f"List up to {count} client difficulties typical for the {data.get('industry', '')} industry. "
        'Return JSON object {"items": [..]} with one item per list entry.'
    )
    return await _suggest_items(prompt, "items")


async def suggest_recent_team_changes(data: dict, count: int = 5) -> list[str]:
    """Suggest examples of recent team changes."""

    prompt = (
        f"Provide up to {count} examples of changes teams might have undergone recently. "
        'Return JSON object {"items": [..]} with one change per list item.'
    )
    return await _suggest_items(prompt, "items")


async def suggest_tech_stack(data: dict, count: int = 5) -> list[str]:
    """Suggest technologies for the tech stack."""

    prompt = (
        f"List up to {count} technologies commonly used in {data.get('industry', '')}. "
        'Return JSON object {"items": [..]} with one technology per list item.'
    )
    return await _suggest_items(prompt, "items")


# ── GPT fill ------------------------------------------------------------------
async def llm_fill(missing_keys: list[str], text: str) -> dict[str, ExtractResult]:
    if not missing_keys:
        return {}

    CHUNK = 40  # keep replies short
    out: dict[str, ExtractResult] = {}
    for i in range(0, len(missing_keys), CHUNK):
        subset = missing_keys[i : i + CHUNK]
        user_msg = (
            f"Extract the following keys and return STRICT JSON only:\n{subset}\n\n"
            f"TEXT:\n```{text[:12_000]}```"
        )
        chat = await client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=500,
            messages=[
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
        )

        raw = safe_json_load(chat.choices[0].message.content or "")
        for k in subset:
            node = raw.get(k, {})
            val = node.get("value") if isinstance(node, dict) else node
            conf = node.get("confidence", 0.5) if isinstance(node, dict) else 0.5
            out[k] = ExtractResult(val, float(conf) if val else 0.0)
    return out


# ── Extraction orchestrator ---------------------------------------------------
async def extract(text: str) -> dict[str, ExtractResult]:
    interim: dict[str, ExtractResult] = {
        k: res
        for k, pat in REGEX_PATTERNS.items()
        if (res := pattern_search(text, k, pat)).value
    }

    if "company_name" not in interim:
        guess = search_company_name(text)
        if guess:
            interim["company_name"] = guess

    if "work_location_city" not in interim:
        guess_city = search_city(text)
        if guess_city:
            interim["work_location_city"] = guess_city
            if "city" not in interim:
                interim["city"] = guess_city

    for k, pat in FALLBACK_PATTERNS.items():
        if k not in interim:
            res = pattern_search(text, k, pat)
            if res.value:
                interim[k] = res

    # salary merge
    if (
        "salary_range" not in interim
        and {"salary_range_min", "salary_range_max"} <= interim.keys()
    ):
        interim["salary_range"] = ExtractResult(
            f"{interim['salary_range_min'].value} – {interim['salary_range_max'].value}",
            min(
                interim["salary_range_min"].confidence,
                interim["salary_range_max"].confidence,
            ),
        )

    missing = [k for k in REGEX_PATTERNS.keys() if k not in interim]
    interim.update(await llm_fill(missing, text))
    return interim


# ── UI helpers ----------------------------------------------------------------
def show_input(
    key: str, default: Any, meta: dict[str, Any], widget_prefix: str = ""
) -> None:
    """Render a widget for the given field.

    Args:
        key: Field name used for session state.
        default: Default value or :class:`ExtractResult`.
        meta: Metadata describing the field.
        widget_prefix: Prefix for the Streamlit widget key.
    """

    field_type = meta.get("field_type", meta.get("field", "text_input"))
    helptext = meta.get("helptext", "")
    required = str(meta.get("is_must", "0")) == "1"
    label = ("★ " if required else "") + meta.get(
        "label", key.replace("_", " ").title()
    )
    # Extract value
    val = getattr(default, "value", default)

    # Field logic
    widget_key = f"{widget_prefix}_{key}" if widget_prefix else key
    used = st.session_state.setdefault("_used_widget_keys", set())
    if widget_key in used:
        suffix = 1
        candidate = f"{widget_key}_{suffix}"
        while candidate in used:
            suffix += 1
            candidate = f"{widget_key}_{suffix}"
        widget_key = candidate
    used.add(widget_key)
    if field_type == "text_area":
        val = st.text_area(
            label,
            value=val or "",
            help=helptext,
            key=widget_key,
            height=100,
        )

    elif field_type == "selectbox":
        options = meta.get("options", []) or []
        if not required:
            options = ["Select…", *options]
        index = options.index(val) if val in options else 0
        val = st.selectbox(
            label,
            options=options,
            index=index,
            key=widget_key,
            help=helptext,
        )
        if not required and val == "Select…":
            val = ""

    elif field_type == "multiselect":
        options = meta.get("options", []) or []
        val = st.multiselect(
            label,
            options=options,
            default=[v for v in (val or []) if v in options],
            key=widget_key,
            help=helptext,
        )

    elif field_type == "number_input":
        try:
            # Try to convert directly
            numeric_val = float(val)
        except (ValueError, TypeError):
            # Try to extract a number from a string like 'Count: 2'
            match = re.search(r"[-+]?\d*\.\d+|\d+", str(val))
            numeric_val = float(match.group()) if match else 0.0

        val = st.number_input(label, value=numeric_val, key=widget_key, help=helptext)

    elif field_type == "date_input":
        try:
            date_val = dateparser.parse(str(val)).date() if val else dt.date.today()
        except Exception:
            date_val = dt.date.today()
        val = st.date_input(label, value=date_val, key=widget_key, help=helptext)

    elif field_type == "checkbox":
        val = st.checkbox(
            label, value=str(val).lower() == "true", key=widget_key, help=helptext
        )

    else:
        val = st.text_input(label, value=val or "", key=widget_key, help=helptext)

    # Save to session state
    st.session_state["data"][key] = val


def value_missing(key: str) -> bool:
    """Return ``True`` if no meaningful value is stored for ``key``."""

    val = st.session_state.get("data", {}).get(key)
    if val is None:
        return True
    if isinstance(val, str):
        return not val.strip()
    if isinstance(val, list):
        return len(val) == 0
    return False


def show_missing(
    key: str,
    extr: dict[str, ExtractResult],
    meta_map: dict[str, Any],
    step_name: str,
) -> None:
    """Display the input only when no value exists."""

    if value_missing(key):
        show_input(
            key, extr.get(key, ExtractResult()), meta_map[key], widget_prefix=step_name
        )


def display_extracted_values_editable(
    extracted: dict[str, ExtractResult], keys: list[str], step_name: str
) -> None:
    """Show extracted values in a compact editable table."""

    rows = []
    for k in keys:
        res = extracted.get(k)
        if res and getattr(res, "value", None):
            rows.append(
                {"_key": k, "Feld": k.replace("_", " ").title(), "Wert": res.value}
            )

    if not rows:
        st.info("Keine Werte extrahiert.")
        return

    df = pd.DataFrame(rows).drop(columns="_key")
    edited = st.data_editor(
        df,
        key=f"{step_name}_extr_editor",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "Feld": st.column_config.TextColumn(disabled=True),
        },
    )

    for i, row in edited.iterrows():
        key = rows[cast(int, i)]["_key"]
        ss["data"][key] = row["Wert"]


def display_missing_inputs(
    step_name: str, meta_fields: list[dict[str, str]], extracted: dict
) -> None:
    """Show prominent inputs for fields without extracted values."""

    missing = [
        m
        for m in meta_fields
        if not (
            m["key"] in extracted and getattr(extracted.get(m["key"]), "value", None)
        )
    ]

    if not missing:
        return

    st.subheader("Fehlende Angaben")
    for meta in missing:
        k = meta["key"]
        result = ExtractResult(ss["data"].get(k), 1.0)
        show_input(k, result, meta, widget_prefix=f"missing_{step_name}")


def display_missing_company_inputs(
    meta_fields: list[dict[str, str]], extracted: dict
) -> None:
    """Special layout for missing company and team data."""

    missing = {
        m["key"]: m
        for m in meta_fields
        if not (
            m["key"] in extracted and getattr(extracted.get(m["key"]), "value", None)
        )
    }

    if not missing:
        return

    company = ss.get("data", {}).get("company_name", "the company")
    col_a, col_b = st.columns(2)

    col_a_keys = [
        "company_size",
        "headquarters_location",
        "industry",
        "brand_name",
        "culture_notes",
        "office_type",
        "office_language",
    ]
    col_b_keys = [
        "department_name",
        "team_size",
        "team_structure",
        "tech_stack",
        "team_challenges",
        "client_difficulties",
        "main_stakeholders",
        "team_motivation",
        "recent_team_changes",
    ]

    with col_a:
        st.markdown(f"### Missing Data on {company}")
        for key in col_a_keys:
            if key in missing:
                meta = missing[key]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta, widget_prefix="missing_COMPANY")

    with col_b:
        st.markdown("### Missing Data on the Department and Team")
        for key in col_b_keys:
            if key in missing:
                meta = missing[key]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta, widget_prefix="missing_COMPANY")


def display_missing_basic_inputs(
    meta_fields: list[dict[str, str]], extracted: dict
) -> None:
    """Special layout for missing inputs in the BASIC step."""

    missing = {
        m["key"]
        for m in meta_fields
        if not (
            m["key"] in extracted and getattr(extracted.get(m["key"]), "value", None)
        )
    }

    if not missing:
        return

    st.subheader("Provide Information on missing basic data")
    meta_map = {m["key"]: m for m in meta_fields}
    col_a_keys = [
        "date_of_employment_start",
        "employment_type",
        "work_location_city",
        "pay_frequency",
        "salary_currency",
        "salary_range",
    ]
    col_b_keys = ["company_name", "department_name", "company_website"]
    col_a, col_b = st.columns(2)
    for key in col_a_keys:
        if key in missing and key in meta_map:
            with col_a:
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta_map[key], widget_prefix="missing_BASIC")
    for key in col_b_keys:
        if key in missing and key in meta_map:
            with col_b:
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta_map[key], widget_prefix="missing_BASIC")


def display_summary() -> None:
    """Show all collected data grouped by step with inline editing."""
    for step_name in ORDER:
        if step_name not in SCHEMA:
            continue
        with st.expander(STEP_TITLES.get(step_name, step_name.title()), expanded=False):
            for meta in SCHEMA[step_name]:
                key = meta["key"]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta, widget_prefix=f"summary_{step_name}")


def display_summary_overview() -> None:
    """Render the most important fields in four columns."""

    def val(field: str) -> str:
        return str(ss["data"].get(field, ""))

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("### Vacancy Details")
        st.write(f"**Job Title:** {val('job_title')}")
        st.write(f"**Company Name:** {val('company_name')}")
        st.write(f"**Start Date Target:** {val('date_of_employment_start')}")
        st.write(f"**Place of Work:** {val('place_of_work')}")
        st.write(f"**Contract Type:** {val('contract_type')}")
        st.write(f"**Work Schedule:** {val('work_schedule')}")

    with col2:
        st.markdown("### About the Role")
        st.write(f"**Role Description:** {val('role_description')}")
        st.write(f"**Task List:** {val('task_list')}")
        st.write(f"**Technical Tasks:** {val('technical_tasks')}")
        st.write(f"**Managerial Tasks:** {val('managerial_tasks')}")
        st.write(f"**Role Keywords:** {val('role_keywords')}")
        st.write(f"**Ideal Candidate:** {val('ideal_candidate_profile')}")
        st.write(f"**Target Industries:** {val('target_industries')}")

    with col3:
        st.markdown("### Skills")
        st.write(f"**Hard Skills:** {val('hard_skills')}")
        st.write(f"**Must Have Skills:** {val('must_have_skills')}")
        st.write(f"**Nice to Have Skills:** {val('nice_to_have_skills')}")
        st.write(f"**Soft Skills:** {val('soft_skills')}")
        st.write(f"**Certifications Required:** {val('certifications_required')}")
        st.write(f"**Domain Expertise:** {val('domain_expertise')}")
        st.write(f"**Language Requirements:** {val('language_requirements')}")
        st.write(f"**Languages Optional:** {val('languages_optional')}")
        st.write(f"**Industry Experience:** {val('industry_experience')}")

    with col4:
        st.markdown("### Benefits")
        st.write(f"**Salary Range (EUR):** {val('salary_range')}")
        st.write(f"**Variable Comp:** {val('variable_comp')}")
        st.write(f"**Vacation Days:** {val('vacation_days')}")
        st.write(f"**Remote Policy:** {val('remote_policy')}")
        st.write(f"**Flexible Hours:** {val('flexible_hours')}")
        st.write(f"**Childcare Support:** {val('childcare_support')}")
        st.write(f"**Learning Budget (EUR):** {val('learning_budget')}")
        st.write(f"**Other Perks:** {val('other_perks')}")
        if ss.get("benefit_list"):
            st.write(f"**Selected Benefits:** {', '.join(ss['benefit_list'])}")


def display_salary_plot() -> None:
    """Show an interactive salary contribution chart."""

    factors = ["job_title", "role_description", "task_list", "location", "skills"]
    selected = st.multiselect("Factors", factors, default=factors)
    job_title_val = ss["data"].get("job_title") if "job_title" in selected else None
    role_desc_val = (
        ss["data"].get("role_description") if "role_description" in selected else None
    )
    task_list_val = ss["data"].get("task_list") if "task_list" in selected else None
    location_val = ss["data"].get("city") if "location" in selected else None
    skills_val = (
        parse_skill_list(ss["data"].get("must_have_skills"))
        if "skills" in selected
        else None
    )
    total, parts = predict_annual_salary(
        cast(str | None, job_title_val),
        cast(str | None, role_desc_val),
        cast(str | None, task_list_val),
        cast(str | None, location_val),
        cast(list[str] | None, skills_val),
    )
    fig = px.bar(
        x=list(parts.keys()),
        y=list(parts.values()),
        labels={"x": "Component", "y": "Contribution"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.write(f"**Expected Annual Salary:** {total} €")
    st.write(
        "This estimate derives from typical market rates, the complexity of the role, location factors, the number of tasks and the required skills. "
        "Each component contributes additively to the final figure. "
        "Data scientist roles or similar often command premiums reflected in the job_title factor. "
        "Longer task lists and detailed role descriptions increase expectations as they imply broader responsibilities. "
        "Locations with high living costs push salaries upwards, while more skills signify higher expertise and thus higher pay."
    )


def display_interview_section(
    meta_fields: list[dict[str, str]], extr: dict[str, ExtractResult]
) -> None:
    """Show interview contacts and process information."""

    def val(field: str) -> str:
        return str(ss["data"].get(field, ""))

    meta_map = {m["key"]: m for m in meta_fields}

    options = ["Receive CVs", "Receive IV-Invites", "Receive offer"]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Line Manager")
        st.write(f"**Name:** {val('line_manager_name')}")
        st.write(f"**Email:** {val('line_manager_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("line_manager_involve", []),
            key="line_manager_involve",
        )
    with col2:
        st.markdown("### HR Contact")
        st.write(f"**Name:** {val('hr_poc_name')}")
        st.write(f"**Email:** {val('hr_poc_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("hr_poc_involve", []),
            key="hr_poc_involve",
        )
    with col3:
        st.markdown("### Finance Contact")
        st.write(f"**Name:** {val('finance_poc_name')}")
        st.write(f"**Email:** {val('finance_poc_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("finance_poc_involve", []),
            key="finance_poc_involve",
        )

    st.subheader("Application Contact")
    app_cols = st.columns(2)
    with app_cols[0]:
        show_input(
            "recruitment_contact_email",
            extr.get("recruitment_contact_email", ExtractResult()),
            meta_map["recruitment_contact_email"],
            widget_prefix="INTERVIEW",
        )
    with app_cols[1]:
        show_input(
            "recruitment_contact_phone",
            extr.get("recruitment_contact_phone", ExtractResult()),
            meta_map["recruitment_contact_phone"],
            widget_prefix="INTERVIEW",
        )

    st.subheader("Interview Process")
    proc_cols = st.columns(2)
    with proc_cols[0]:
        show_input(
            "recruitment_steps",
            extr.get("recruitment_steps", ExtractResult()),
            meta_map["recruitment_steps"],
            widget_prefix="INTERVIEW",
        )
    with proc_cols[1]:
        show_input(
            "recruitment_timeline",
            extr.get("recruitment_timeline", ExtractResult()),
            meta_map["recruitment_timeline"],
            widget_prefix="INTERVIEW",
        )

    cols = st.columns(2)
    with cols[0]:
        show_input(
            "number_of_interviews",
            extr.get("number_of_interviews", ExtractResult()),
            meta_map["number_of_interviews"],
            widget_prefix="INTERVIEW",
        )
    with cols[1]:
        show_input(
            "interview_format",
            extr.get("interview_format", ExtractResult()),
            meta_map["interview_format"],
            widget_prefix="INTERVIEW",
        )

    st.subheader("Onboarding & Probation")
    onboard_cols = st.columns(3)
    with onboard_cols[0]:
        show_input(
            "probation_period",
            extr.get("probation_period", ExtractResult()),
            meta_map["probation_period"],
            widget_prefix="INTERVIEW",
        )
    with onboard_cols[1]:
        show_input(
            "mentorship_program",
            extr.get("mentorship_program", ExtractResult()),
            meta_map["mentorship_program"],
            widget_prefix="INTERVIEW",
        )
    with onboard_cols[2]:
        show_input(
            "welcome_package",
            extr.get("welcome_package", ExtractResult()),
            meta_map["welcome_package"],
            widget_prefix="INTERVIEW",
        )

    show_input(
        "onboarding_process",
        extr.get("onboarding_process", ExtractResult()),
        meta_map["onboarding_process"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "onboarding_process_overview",
        extr.get("onboarding_process_overview", ExtractResult()),
        meta_map["onboarding_process_overview"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "interview_stage_count",
        extr.get("interview_stage_count", ExtractResult()),
        meta_map["interview_stage_count"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "interview_docs_required",
        extr.get("interview_docs_required", ExtractResult()),
        meta_map["interview_docs_required"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "assessment_tests",
        extr.get("assessment_tests", ExtractResult()),
        meta_map["assessment_tests"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "interview_notes",
        extr.get("interview_notes", ExtractResult()),
        meta_map["interview_notes"],
        widget_prefix="INTERVIEW",
    )
    show_input(
        "application_instructions",
        extr.get("application_instructions", ExtractResult()),
        meta_map["application_instructions"],
        widget_prefix="INTERVIEW",
    )


img_path = Path("images/AdobeStock_506577005.jpeg")


# Bild als Base64 laden (damit es im CSS eingebettet werden kann)
def get_base64_image(img_path):
    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


# CSS-Block für halbtransparentes Hintergrundbild
def set_background(image_path: Path, opacity=0.5):
    img_url = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(rgba(255, 255, 255, {1-opacity}), rgba(255, 255, 255, {0-opacity})), 
                        url("{img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
    """,
        unsafe_allow_html=True,
    )


# Hintergrund aktivieren
set_background(img_path, opacity=0.5)


# Mapping für Subtitles pro Step
STEP_SUBTITLES = {
    "BASIC": (
        "Hier werden die Basisdaten zur Vakanz gesammelt – sie sind wichtig für die spätere Zuordnung, Suche und Vergleichbarkeit. "
        "Je vollständiger diese Angaben sind, desto gezielter kann die Stelle gefunden und analysiert werden."
    ),
    "COMPANY": (
        "Informationen zu Unternehmen, Team und Abteilung helfen, die Vakanz besser zu verorten "
        "und passgenaues Employer Branding zu ermöglichen. Solche Angaben erhöhen "
        "die Glaubwürdigkeit und Transparenz gegenüber Kandidat:innen."
    ),
    "ROLE": (
        "Die Rolle bündelt Beschreibung und Aufgaben – hier bitte besonders genau sein. "
        "Je klarer Verantwortlichkeiten, Prioritäten und Aufgaben definiert sind, desto besser passen "
        "die späteren Kandidat:innen."
    ),
    "SKILLS": (
        "An dieser Stelle werden die fachlichen und persönlichen Kompetenzen festgehalten, die für die Vakanz wichtig sind. "
        "Eine genaue Definition der Anforderungen erleichtert das Matching im späteren Prozess."
    ),
    "BENEFITS": (
        "In diesem Abschnitt werden die Vorteile und Benefits präsentiert, die das Unternehmen bietet. "
        "Attraktive Zusatzleistungen steigern die Arbeitgeberattraktivität und fördern Bewerbungen."
    ),
    "INTERVIEW": (
        "Der Interviewprozess und die beteiligten Personen werden in diesem Abschnitt dokumentiert. "
        "Eine klare Struktur des Prozesses sorgt für ein professionelles Kandidaten-Erlebnis."
    ),
    "SUMMARY": (
        "Im letzten Schritt werden alle Informationen noch einmal übersichtlich zusammengefasst. "
        "Überprüfe die Angaben und exportiere das vollständige Anforderungsprofil."
    ),
}


# AI-Functions


# --- a) Jobad-Generator mit DSGVO, SEO, Edit, PDF ---
async def generate_jobad(data: dict) -> str:
    """
    Generiert eine professionelle, DSGVO-konforme und SEO-optimierte Stellenanzeige auf Basis der gesammelten Daten.
    Rückgabe: Jobad als Markdown/Text.
    """
    prompt = (
        "Erstelle eine vollständige, DSGVO-konforme und suchmaschinenoptimierte Stellenanzeige "
        "auf Basis der folgenden strukturierten Daten. "
        "Achte auf Transparenz (Datenschutz), genderneutrale Sprache und passende Formatierung. "
        f"Daten: {json.dumps(data, ensure_ascii=False, default=str)}"
    )
    chat = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.7,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": "Du bist HR- und SEO-Textexperte."},
            {"role": "user", "content": prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()


def download_as_pdf(text: str, filename: str = "jobad.pdf"):
    """Convert text to a downloadable PDF."""
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    for line in text.splitlines():
        pdf.cell(0, 10, text=line, new_y="NEXT")
    pdf.output(filename)
    with open(filename, "rb") as f:
        st.download_button(
            "Download PDF", f, file_name=filename, mime="application/pdf"
        )


# --- b) Interview-Vorbereitungs-Sheet ---
async def generate_interview_sheet(data: dict) -> str:
    """
    Erstellt ein kompaktes, tabellarisches Vorbereitungsblatt für Line und HR auf Basis der wichtigsten Anforderungen, Aufgaben und Wunschkriterien.
    Rückgabe: Markdown- oder HTML-Tabelle.
    """
    prompt = (
        "Prepare an interview briefing for the hiring team. "
        "Tabulate key role criteria, must-have vs nice-to-have skills, and suggested interview questions based on the role profile.\n"
        f"Profile Data: {json.dumps(data, ensure_ascii=False, default=str)}"
    )
    system_msg = "You are an expert interview coach for HR and hiring managers."
    return await generate_text(
        prompt,
        model="gpt-4o",
        temperature=0.5,
        max_tokens=800,
        system_msg=system_msg,
    )


# --- c) Boolean Searchstring ---
async def generate_boolean_search(data: dict) -> str:
    """
    Erstellt einen professionellen, auf die Vakanz optimierten Boolean Searchstring für Jobbörsen, LinkedIn, XING etc.
    """
    prompt = (
        "Create a concise, optimized Boolean search string for this vacancy, "
        "using key responsibilities, requirements, and skills as the basis.\n"
        f"Role Profile: {json.dumps(data, ensure_ascii=False, default=str)}"
    )
    system_msg = "You are an expert in talent sourcing and boolean search."
    return await generate_text(
        prompt,
        model="gpt-4o",
        temperature=0.3,
        max_tokens=300,
        system_msg=system_msg,
    )


# --- d) Arbeitsvertrag-Generator ---
async def generate_contract(data: dict) -> str:
    """
    Erstellt einen einfachen Entwurf für einen Arbeitsvertrag auf Basis der extrahierten Stammdaten.
    (Beachte: Dies ist keine Rechtsberatung und ersetzt keinen Juristen!)
    """
    prompt = (
        "Erstelle einen Muster-Arbeitsvertrag (nur als Vorlage, keine Rechtsberatung!) auf Basis dieser strukturierten Daten. "
        "Inkludiere alle relevanten Pflichtangaben (Name, Stelle, Vergütung, Beginn, Probezeit, Aufgaben, Arbeitszeit). "
        f"Daten: {json.dumps(data, ensure_ascii=False, default=str)}"
    )
    chat = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.4,
        max_tokens=1200,
        messages=[
            {
                "role": "system",
                "content": "Du bist Vertragsgenerator für HR (keine Rechtsberatung).",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()


def estimate_salary_range(job_title: str, seniority: str) -> str:
    """Estimate salary range (per year in EUR) based on job title and seniority level."""

    base: float = 50000
    title = job_title.lower()
    if "data scientist" in title:
        base = 80000
    elif "software engineer" in title or "developer" in title:
        base = 75000
    elif "project manager" in title:
        base = 90000

    level = seniority.lower() if seniority else ""
    if "junior" in level:
        base *= 0.8
    elif "senior" in level:
        base *= 1.2
    elif "lead" in level or "head" in level:
        base *= 1.5

    lower = int(base * 0.9)
    upper = int(base * 1.1)
    return f"{lower}–{upper} €"


def calculate_total_compensation(
    salary_range: tuple[int, int], benefits: list[str]
) -> int:
    """Rough calculation of annual compensation cost given salary range and selected benefits."""

    benefit_costs = {
        "Health Insurance": 500,
        "Company Car": 5000,
        "Flexible Hours": 100,
        "Home Office Options": 200,
        "Training Budget": 800,
        "Pension Plan": 1000,
    }

    extra_cost = sum(benefit_costs.get(b, 0) for b in benefits)
    total = (salary_range[1] if salary_range else 0) + extra_cost
    return total


def predict_annual_salary(
    job_title: str | None,
    role_description: str | None,
    task_list: str | None,
    location: str | None,
    skills: list[str] | None,
) -> tuple[int, dict[str, int]]:
    """Estimate annual salary and contribution of each component."""

    base = 30000
    contrib = {
        "job_title": 0,
        "role_description": 0,
        "task_list": 0,
        "location": 0,
        "skills": 0,
    }

    if job_title:
        title = job_title.lower()
        if "data scientist" in title:
            contrib["job_title"] += 30000
        elif "engineer" in title or "developer" in title:
            contrib["job_title"] += 25000
        elif "manager" in title:
            contrib["job_title"] += 20000
        else:
            contrib["job_title"] += 15000

    if role_description:
        contrib["role_description"] += min(10000, len(role_description.split()) * 20)

    if task_list:
        contrib["task_list"] += min(8000, len(task_list.split()) * 15)

    if location:
        if location.lower() in ["berlin", "munich", "hamburg", "frankfurt"]:
            contrib["location"] += 5000
        else:
            contrib["location"] += 2000

    if skills:
        contrib["skills"] += len(skills) * 1000

    total = base + sum(contrib.values())
    return total, contrib


# ── Streamlit main ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="🧭 Recruitment Need Analysis Tool",
        page_icon="🧭",
        layout="wide",
    )

    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("data", {})
    ss.setdefault("extracted", {})
    ss.setdefault("benefit_list", [])

    def goto(i: int):
        ss["step"] = i

    step = ss["step"]

    # ----------- 0: Welcome / Upload-Page -----------
    if step == 0:
        # Schönes Welcome-Design!
        with open("images/color1_logo_transparent_background.png", "rb") as img_file:
            logo_b64 = base64.b64encode(img_file.read()).decode()

        st.markdown(
            f"""
        <div style="position:relative;">
            <img src="data:image/png;base64,{logo_b64}" style="position:absolute; top:0; right:0; width:200px;" />
            <div class="black-text" style="text-align:center;">
                <h2 style="font-size:26pt;">Recruitment Need Analysis 🧭</h2>
                <p>Welcome! This Tool helps you quickly create a complete vacancy profile.</p>
                <p>Upload a Job Advert. All relevant information will be extracted and preprocessed automatically.</p>
                <p>Afterwards, start discovering missing data in your Specification in order to Minimise Costs and to ensure Maximum Recruitment Success .</p>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.divider()
        job_title_default = ss["data"].get("job_title")
        if not job_title_default:
            extr_title = ss.get("extracted", {}).get("job_title")
            if isinstance(extr_title, ExtractResult):
                job_title_default = extr_title.value or ""

        if not st.session_state.get("job_title") and job_title_default:
            st.session_state.job_title = job_title_default

        up = None
        extract_btn = False
        left_pad, center_col, right_pad = st.columns([15, 70, 15])

        with center_col:
            st.text_input(
                "Job Title",
                value=job_title_default or "",
                key="job_title",
                label_visibility="visible",
            )

            up = st.file_uploader(
                "Upload Job Description (PDF or DOCX)",
                type=["pdf", "docx"],
            )

            st.markdown(
                "<p style='text-align:center; font-size:calc(1rem + 2pt);'>Start discovering missing data in your specification in order to minimise Costs and to ensure maximum recruitment Success</p>",
                unsafe_allow_html=True,
            )

            btn_left, btn_right = st.columns(2)
            with btn_left:
                extract_btn = st.button(
                    "Extract Vacancy Data",
                    disabled=not up,
                    use_container_width=True,
                )
            with btn_right:
                st.button(
                    "Next →",
                    on_click=lambda: goto(1),
                    use_container_width=True,
                )

        ss["data"]["job_title"] = st.session_state.job_title
        if st.session_state.job_title and not ss.get("extracted", {}).get(
            "BASIC", {}
        ).get("job_title"):
            ss.setdefault("extracted", {}).setdefault("BASIC", {})["job_title"] = (
                ExtractResult(st.session_state.job_title, 1.0)
            )

        if extract_btn and up:
            with st.spinner("Extracting…"):
                file_bytes = up.read()
                text = extract_text_from_file(file_bytes, up.type)
                flat = asyncio.run(extract(text))
                ss["extracted"] = group_by_step(flat)
                title_res = ss["extracted"].get("BASIC", {}).get("job_title")
                if isinstance(title_res, ExtractResult) and title_res.value:
                    ss["data"]["job_title"] = title_res.value
                st.rerun()
    # ----------- 1..n: Wizard -----------
    elif 1 <= step < len(STEPS) + 1:
        step_idx = step - 1
        step_name = ORDER[step_idx]
        meta_fields = SCHEMA[step_name]  # <-- Zuerst setzen!
        fields = [item["key"] for item in meta_fields]
        extr: dict[str, ExtractResult] = ss.get("extracted", {}).get(step_name, {})

        # Headline & Subtitle
        title = STEP_TITLES.get(step_name, step_name.title())
        if step_name == "BENEFITS":
            job_title_val = ss.get("data", {}).get("job_title", "")
            if job_title_val:
                title = f"{title} – {job_title_val}"
        elif step_name == "BASIC":
            job_title_val = ss.get("data", {}).get("job_title", "")
            if job_title_val:
                title = f"Basic Data on your {job_title_val}-Vacancy"
            else:
                title = "Basic Data on your Vacancy"
        st.markdown(
            f"<h2 style='text-align:center'>{title}</h2>",
            unsafe_allow_html=True,
        )
        subtitle = STEP_SUBTITLES.get(step_name, "")
        if subtitle:
            st.markdown(
                f"<div style='text-align:center; color:#bbb; margin-bottom:24px'>{subtitle}</div>",
                unsafe_allow_html=True,
            )

        if step_name == "INTERVIEW":
            display_interview_section(meta_fields, extr)

        # Extrahierte Werte kompakt darstellen
        display_extracted_values_editable(extr, fields, step_name)

        # Prominent fehlende Felder abfragen
        if step_name == "BASIC":
            display_missing_basic_inputs(meta_fields, extr)
        elif step_name == "COMPANY":
            display_missing_company_inputs(meta_fields, extr)
        else:
            display_missing_inputs(step_name, meta_fields, extr)

        if step_name == "BASIC":
            meta_map = {m["key"]: m for m in meta_fields}
            if value_missing("job_title"):
                show_input(
                    "job_title",
                    extr.get("job_title", ExtractResult()),
                    meta_map["job_title"],
                    widget_prefix=step_name,
                )

            cols = st.columns(2)
            with cols[0]:
                if value_missing("contract_type"):
                    show_input(
                        "contract_type",
                        extr.get("contract_type", ExtractResult()),
                        meta_map["contract_type"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("work_schedule"):
                    show_input(
                        "work_schedule",
                        extr.get("work_schedule", ExtractResult()),
                        meta_map["work_schedule"],
                        widget_prefix=step_name,
                    )
                if (
                    not value_missing("work_schedule")
                    and ss.get("data", {}).get("work_schedule") == "Hybrid"
                ):
                    default_pct = int(ss.get("data", {}).get("onsite_percentage", 50))
                    pct = st.slider(
                        "% Onsite vs Remote",
                        min_value=0,
                        max_value=100,
                        value=default_pct,
                        step=5,
                        key="onsite_percentage",
                    )
                    ss["data"]["onsite_percentage"] = pct
        elif step_name == "COMPANY":
            meta_map = {m["key"]: m for m in meta_fields}
            cols = st.columns(2)
            with cols[0]:
                if value_missing("company_name"):
                    show_input(
                        "company_name",
                        extr.get("company_name", ExtractResult()),
                        meta_map["company_name"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("city"):
                    show_input(
                        "city",
                        extr.get("city", ExtractResult()),
                        meta_map["city"],
                        widget_prefix=step_name,
                    )

            cols = st.columns(2)
            with cols[0]:
                if value_missing("industry"):
                    show_input(
                        "industry",
                        extr.get("industry", ExtractResult()),
                        meta_map["industry"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("company_size"):
                    show_input(
                        "company_size",
                        extr.get("company_size", ExtractResult()),
                        meta_map["company_size"],
                        widget_prefix=step_name,
                    )

            cols = st.columns(2)
            with cols[0]:
                if value_missing("headquarters_location"):
                    show_input(
                        "headquarters_location",
                        extr.get("headquarters_location", ExtractResult()),
                        meta_map["headquarters_location"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("place_of_work"):
                    show_input(
                        "place_of_work",
                        extr.get("place_of_work", ExtractResult()),
                        meta_map["place_of_work"],
                        widget_prefix=step_name,
                    )

            cols = st.columns(2)
            with cols[0]:
                if value_missing("department_name"):
                    show_input(
                        "department_name",
                        extr.get("department_name", ExtractResult()),
                        meta_map["department_name"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("team_size"):
                    show_input(
                        "team_size",
                        extr.get("team_size", ExtractResult()),
                        meta_map["team_size"],
                        widget_prefix=step_name,
                    )

            cols = st.columns(2)
            with cols[0]:
                if value_missing("company_website"):
                    show_input(
                        "company_website",
                        extr.get("company_website", ExtractResult()),
                        meta_map["company_website"],
                        widget_prefix=step_name,
                    )
            with cols[1]:
                if value_missing("brand_name"):
                    show_input(
                        "brand_name",
                        extr.get("brand_name", ExtractResult()),
                        meta_map["brand_name"],
                        widget_prefix=step_name,
                    )

            if value_missing("team_structure"):
                show_input(
                    "team_structure",
                    extr.get("team_structure", ExtractResult()),
                    meta_map["team_structure"],
                    widget_prefix=step_name,
                )

            if value_missing("reports_to"):
                show_input(
                    "reports_to",
                    extr.get("reports_to", ExtractResult()),
                    meta_map["reports_to"],
                    widget_prefix=step_name,
                )

            with st.expander("Team & Culture Context", expanded=False):
                cols = st.columns([4, 1])
                with cols[0]:
                    show_input(
                        "tech_stack",
                        extr.get("tech_stack", ExtractResult()),
                        meta_map["tech_stack"],
                        widget_prefix=step_name,
                    )
                with cols[1]:
                    if st.button("Generate Ideas", key="gen_tech_stack"):
                        with st.spinner("Generiere…"):
                            try:
                                ss["tech_stack_suggestions"] = asyncio.run(
                                    suggest_tech_stack(ss["data"])
                                )
                            except Exception as e:
                                logging.error("tech stack suggestion failed: %s", e)
                                ss["tech_stack_suggestions"] = []
                sel_ts = selectable_buttons(
                    ss.get("tech_stack_suggestions", []),
                    "",
                    "sel_tech_stack",
                    cols=2,
                )
                current_ts = parse_skill_list(ss["data"].get("tech_stack"))
                for s in sel_ts:
                    if s not in current_ts:
                        current_ts.append(s)
                ss["data"]["tech_stack"] = ", ".join(current_ts)

                if value_missing("culture_notes"):
                    show_input(
                        "culture_notes",
                        extr.get("culture_notes", ExtractResult()),
                        meta_map["culture_notes"],
                        widget_prefix=step_name,
                    )

                cols = st.columns([4, 1])
                with cols[0]:
                    if value_missing("team_challenges"):
                        show_input(
                            "team_challenges",
                            extr.get("team_challenges", ExtractResult()),
                            meta_map["team_challenges"],
                            widget_prefix=step_name,
                        )
                with cols[1]:
                    if st.button("Generate Ideas", key="gen_team_challenges"):
                        with st.spinner("Generiere…"):
                            try:
                                ss["team_challenges_suggestions"] = asyncio.run(
                                    suggest_team_challenges(ss["data"])
                                )
                            except Exception as e:
                                logging.error("team challenge suggestion failed: %s", e)
                                ss["team_challenges_suggestions"] = []
                sel_tc = selectable_buttons(
                    ss.get("team_challenges_suggestions", []),
                    "",
                    "sel_team_challenges",
                    cols=2,
                )
                cur_tc = parse_skill_list(ss["data"].get("team_challenges"))
                for s in sel_tc:
                    if s not in cur_tc:
                        cur_tc.append(s)
                ss["data"]["team_challenges"] = ", ".join(cur_tc)

                cols = st.columns([4, 1])
                with cols[0]:
                    if value_missing("client_difficulties"):
                        show_input(
                            "client_difficulties",
                            extr.get("client_difficulties", ExtractResult()),
                            meta_map["client_difficulties"],
                            widget_prefix=step_name,
                        )
                with cols[1]:
                    if st.button("Generate Ideas", key="gen_client_difficulties"):
                        with st.spinner("Generiere…"):
                            try:
                                ss["client_difficulties_suggestions"] = asyncio.run(
                                    suggest_client_difficulties(ss["data"])
                                )
                            except Exception as e:
                                logging.error(
                                    "client difficulty suggestion failed: %s", e
                                )
                                ss["client_difficulties_suggestions"] = []
                sel_cd = selectable_buttons(
                    ss.get("client_difficulties_suggestions", []),
                    "",
                    "sel_client_difficulties",
                    cols=2,
                )
                cur_cd = parse_skill_list(ss["data"].get("client_difficulties"))
                for s in sel_cd:
                    if s not in cur_cd:
                        cur_cd.append(s)
                ss["data"]["client_difficulties"] = ", ".join(cur_cd)

                if value_missing("main_stakeholders"):
                    show_input(
                        "main_stakeholders",
                        extr.get("main_stakeholders", ExtractResult()),
                        meta_map["main_stakeholders"],
                        widget_prefix=step_name,
                    )

                if value_missing("team_motivation"):
                    show_input(
                        "team_motivation",
                        extr.get("team_motivation", ExtractResult()),
                        meta_map["team_motivation"],
                        widget_prefix=step_name,
                    )

                cols = st.columns([4, 1])
                with cols[0]:
                    if value_missing("recent_team_changes"):
                        show_input(
                            "recent_team_changes",
                            extr.get("recent_team_changes", ExtractResult()),
                            meta_map["recent_team_changes"],
                            widget_prefix=step_name,
                        )
                with cols[1]:
                    if st.button("Generate Ideas", key="gen_recent_team_changes"):
                        with st.spinner("Generiere…"):
                            try:
                                ss["recent_team_changes_suggestions"] = asyncio.run(
                                    suggest_recent_team_changes(ss["data"])
                                )
                            except Exception as e:
                                logging.error("recent changes suggestion failed: %s", e)
                                ss["recent_team_changes_suggestions"] = []
                sel_rc = selectable_buttons(
                    ss.get("recent_team_changes_suggestions", []),
                    "",
                    "sel_recent_team_changes",
                    cols=2,
                )
                cur_rc = parse_skill_list(ss["data"].get("recent_team_changes"))
                for s in sel_rc:
                    if s not in cur_rc:
                        cur_rc.append(s)
                ss["data"]["recent_team_changes"] = ", ".join(cur_rc)

            if value_missing("office_language"):
                show_input(
                    "office_language",
                    extr.get("office_language", ExtractResult()),
                    meta_map["office_language"],
                    widget_prefix=step_name,
                )
            if value_missing("office_type"):
                show_input(
                    "office_type",
                    extr.get("office_type", ExtractResult()),
                    meta_map["office_type"],
                    widget_prefix=step_name,
                )

        elif step_name == "ROLE":
            meta_map = {m["key"]: m for m in meta_fields}

            st.subheader("Role Summary")
            show_missing("role_description", extr, meta_map, step_name)
            cols = st.columns(2)
            with cols[0]:
                show_missing("role_type", extr, meta_map, step_name)
            with cols[1]:
                show_missing("role_keywords", extr, meta_map, step_name)

            st.subheader("Responsibilities")
            show_missing("primary_responsibilities", extr, meta_map, step_name)
            show_missing("key_responsibilities", extr, meta_map, step_name)

            sup_col, direct_col = st.columns(2)
            with sup_col:
                show_missing("supervises", extr, meta_map, step_name)
            with direct_col:
                if ss.get("data", {}).get("supervises"):
                    val = int(ss.get("data", {}).get("direct_reports_count", 0))
                    val = st.number_input(
                        meta_map["direct_reports_count"]["label"],
                        value=val,
                        step=1,
                        format="%d",
                        key=f"{step_name}_direct_reports_count",
                        help=meta_map["direct_reports_count"].get("helptext", ""),
                    )
                    ss["data"]["direct_reports_count"] = int(val)
                else:
                    ss["data"]["direct_reports_count"] = 0

            with st.expander("Projects & Metrics", expanded=False):
                show_missing("role_priority_projects", extr, meta_map, step_name)
                show_missing("key_deliverables", extr, meta_map, step_name)
                show_missing(
                    "role_performance_metrics",
                    extr,
                    meta_map,
                    step_name,
                )
                show_missing("success_metrics", extr, meta_map, step_name)

            st.subheader("Tasks")
            show_missing("task_list", extr, meta_map, step_name)
            with st.expander("Detailed Task Categories", expanded=False):
                show_missing("technical_tasks", extr, meta_map, step_name)
                show_missing("managerial_tasks", extr, meta_map, step_name)
                show_missing("administrative_tasks", extr, meta_map, step_name)
                show_missing("customer_facing_tasks", extr, meta_map, step_name)
                show_missing("internal_reporting_tasks", extr, meta_map, step_name)
                show_missing("performance_tasks", extr, meta_map, step_name)
                show_missing("innovation_tasks", extr, meta_map, step_name)

            st.subheader("Additional Requirements")
            cols = st.columns(3)
            with cols[0]:
                show_missing("task_prioritization", extr, meta_map, step_name)
            with cols[1]:
                show_missing("decision_authority", extr, meta_map, step_name)
            with cols[2]:
                show_missing("process_improvement", extr, meta_map, step_name)

            cols = st.columns(3)
            with cols[0]:
                show_missing("innovation_expected", extr, meta_map, step_name)
            with cols[1]:
                show_missing("on_call", extr, meta_map, step_name)
            with cols[2]:
                show_missing("physical_duties", extr, meta_map, step_name)
            cols = st.columns(3)
            with cols[0]:
                show_missing("travel_required", extr, meta_map, step_name)

        elif step_name == "SKILLS":
            meta_map = {m["key"]: m for m in meta_fields}

            show_missing("seniority_level", extr, meta_map, step_name)

            # Core skills
            show_missing("must_have_skills", extr, meta_map, step_name)
            show_missing("nice_to_have_skills", extr, meta_map, step_name)
            show_missing("hard_skills", extr, meta_map, step_name)
            show_missing("soft_skills", extr, meta_map, step_name)
            show_missing("certifications_required", extr, meta_map, step_name)

            st.subheader("Language Skills")
            show_missing("language_requirements", extr, meta_map, step_name)
            show_missing("languages_optional", extr, meta_map, step_name)

            ind_required = st.checkbox(
                "Industry experience",
                value=bool(ss.get("data", {}).get("industry_experience_required")),
                key=f"{step_name}_industry_exp_req",
            )
            ss["data"]["industry_experience_required"] = ind_required
            if ind_required:
                inds = st.multiselect(
                    "Select industries",
                    INDUSTRY_OPTIONS,
                    default=ss.get("data", {}).get("industry_experience", []),
                    key=f"{step_name}_industry_list",
                )
                ss["data"]["industry_experience"] = inds

            dom_required = st.checkbox(
                "Domain expertise",
                value=bool(ss.get("data", {}).get("domain_expertise_required")),
                key=f"{step_name}_domain_exp_req",
            )
            ss["data"]["domain_expertise_required"] = dom_required
            if dom_required:
                doms = st.multiselect(
                    "Select domains",
                    DOMAIN_OPTIONS,
                    default=ss.get("data", {}).get("domain_expertise", []),
                    key=f"{step_name}_domain_list",
                )
                ss["data"]["domain_expertise"] = doms

            st.subheader("Key Competencies")
            comp_cols = st.columns(2)
            with comp_cols[0]:
                show_missing("analytical_skills", extr, meta_map, step_name)
                show_missing("project_management_skills", extr, meta_map, step_name)
            with comp_cols[1]:
                show_missing("communication_skills", extr, meta_map, step_name)
                show_missing("leadership_competencies", extr, meta_map, step_name)

            st.subheader("Technical Environment")
            show_missing("tool_proficiency", extr, meta_map, step_name)
            show_missing("tech_stack", extr, meta_map, step_name)
            show_missing("it_skills", extr, meta_map, step_name)
            show_missing("soft_requirement_details", extr, meta_map, step_name)

            st.subheader("Other Requirements")
            other_cols = st.columns(2)
            with other_cols[0]:
                years_val = int(ss.get("data", {}).get("years_experience_min", 0))
                years_val = st.number_input(
                    meta_map["years_experience_min"]["label"],
                    value=years_val,
                    step=1,
                    format="%d",
                    key=f"{step_name}_years_experience_min",
                    help=meta_map["years_experience_min"].get("helptext", ""),
                )
                ss["data"]["years_experience_min"] = int(years_val)

        else:
            if step_name == "BENEFITS":
                meta_map = {m["key"]: m for m in meta_fields}

                cols_a, cols_b = st.columns(2)
                with cols_a:
                    show_missing("vacation_days", extr, meta_map, step_name)
                    show_missing("remote_policy", extr, meta_map, step_name)
                    if ss.get("data", {}).get("remote_policy") not in {"", "Onsite"}:
                        pct = int(ss.get("data", {}).get("remote_percentage", 50))
                        pct = st.slider(
                            "% Remote", 0, 100, pct, 5, key="remote_percentage"
                        )
                        ss["data"]["remote_percentage"] = pct

                    show_missing("flexible_hours", extr, meta_map, step_name)
                    if ss.get("data", {}).get("flexible_hours") not in {"", "No"}:
                        txt = st.text_input(
                            "Flexibility Details", key="flexible_hours_details"
                        )
                        ss["data"]["flexible_hours_details"] = txt

                    show_missing("company_car", extr, meta_map, step_name)
                    if ss.get("data", {}).get("company_car"):
                        car = st.text_input("Car Class", key="car_class")
                        ss["data"]["car_class"] = car

                    show_missing("stock_options", extr, meta_map, step_name)
                    if ss.get("data", {}).get("stock_options"):
                        opt = st.text_input(
                            "Stock Option Details", key="stock_options_details"
                        )
                        ss["data"]["stock_options_details"] = opt

                    show_missing("bonus_scheme", extr, meta_map, step_name)
                    if ss.get("data", {}).get("bonus_scheme"):
                        txt = st.text_area(
                            meta_map["commission_structure"]["label"],
                            key="commission_structure",
                        )
                        ss["data"]["commission_structure"] = txt

                with cols_b:
                    show_missing("relocation_support", extr, meta_map, step_name)
                    show_missing("childcare_support", extr, meta_map, step_name)
                    show_missing("learning_budget", extr, meta_map, step_name)
                    show_missing("sabbatical_option", extr, meta_map, step_name)
                    show_missing("health_insurance", extr, meta_map, step_name)
                    show_missing("pension_plan", extr, meta_map, step_name)
                    show_missing("other_perks", extr, meta_map, step_name)
                    if ss.get("data", {}).get("other_perks"):
                        txt = st.text_area(
                            "Other Perks Details", key="other_perks_details"
                        )
                        ss["data"]["other_perks_details"] = txt
                    show_missing("visa_sponsorship", extr, meta_map, step_name)
            else:
                current_cols = 2
                cols = st.columns(current_cols)
                col_idx = 0

                for meta in meta_fields:
                    key = meta["key"]
                    result = extr.get(key) if key in extr else ExtractResult()
                    field_type = meta.get("field_type", meta.get("field", "text_input"))

                    if field_type == "text_area":
                        cols = st.columns(1)
                        with cols[0]:
                            show_input(key, result, meta, widget_prefix=step_name)
                        cols = st.columns(current_cols)
                        col_idx = 0
                        continue

                    needed = 3 if field_type == "checkbox" else 2
                    if needed != current_cols or col_idx >= needed:
                        cols = st.columns(needed)
                        current_cols = needed
                        col_idx = 0

                    with cols[col_idx]:
                        show_input(key, result, meta, widget_prefix=step_name)

                    col_idx += 1

        if step_name == "SKILLS":
            if "hard_skill_suggestions" not in ss:
                with st.spinner("AI analysiert Skills…"):
                    try:
                        ss["hard_skill_suggestions"] = asyncio.run(
                            suggest_hard_skills(ss["data"])
                        )
                        ss["soft_skill_suggestions"] = asyncio.run(
                            suggest_soft_skills(ss["data"])
                        )
                    except Exception as e:
                        logging.error("skill suggestion failed: %s", e)
                        ss["hard_skill_suggestions"] = []
                        ss["soft_skill_suggestions"] = []

            job_title = cast(str | None, ss["data"].get("job_title"))
            if job_title:
                hard_label = f"AI-Suggested Hard Skills for your {job_title} Vacancy"
            else:
                hard_label = "AI-Suggested Hard Skills"
            hard_sel = selectable_buttons(
                ss.get("hard_skill_suggestions", []),
                hard_label,
                "selected_hard_skills",
            )
            current_hard = parse_skill_list(ss["data"].get("hard_skills"))
            for sk in hard_sel:
                if sk not in current_hard:
                    current_hard.append(sk)
            ss["data"]["hard_skills"] = ", ".join(current_hard)

            if job_title:
                soft_label = f"AI-Suggested Soft Skills for your {job_title} Vacancy"
            else:
                soft_label = "AI-Suggested Soft Skills"
            soft_sel = selectable_buttons(
                ss.get("soft_skill_suggestions", []),
                soft_label,
                "selected_soft_skills",
            )
            current_soft = parse_skill_list(ss["data"].get("soft_skills"))
            for sk in soft_sel:
                if sk not in current_soft:
                    current_soft.append(sk)
            ss["data"]["soft_skills"] = ", ".join(current_soft)

        if step_name == "BENEFITS":
            st.subheader("AI Benefit Suggestions")

            def benefit_row(text: str, key: str, func) -> list[str]:
                row = st.columns([1, 1, 6, 1])
                with row[0]:
                    st.markdown("Generate")
                with row[1]:
                    count = st.number_input(
                        label="",
                        min_value=1,
                        max_value=10,
                        value=5,
                        step=1,
                        key=f"count_{key}",
                    )
                with row[2]:
                    st.markdown(text)
                with row[3]:
                    if st.button("Generate", key=f"gen_{key}"):
                        with st.spinner("Generiere…"):
                            try:
                                ss[key] = asyncio.run(func(ss["data"], int(count)))
                            except Exception as e:
                                logging.error("benefit suggestion failed: %s", e)
                                ss[key] = []
                return selectable_buttons(ss.get(key, []), "", f"sel_{key}")

            title = ss["data"].get("job_title", "this role")
            company = ss["data"].get("company_name", "your company")

            sel_title = benefit_row(
                f"Benefits for {title}",
                "benefit_title",
                suggest_benefits_by_title,
            )
            sel_loc = benefit_row(
                "local Benefits",
                "benefit_loc",
                suggest_benefits_by_location,
            )
            sel_comp = benefit_row(
                f"Benefits offered by the competitors of {company}",
                "benefit_comp",
                suggest_benefits_competitors,
            )

            ss["benefit_list"] = list({*sel_title, *sel_loc, *sel_comp})

        prev, nxt = st.columns(2)
        prev.button("← Back", disabled=step == 1, on_click=lambda: goto(step - 1))
        required_keys = [
            meta["key"] for meta in meta_fields if meta.get("is_must", "0") == "1"
        ]
        ok = all(ss["data"].get(k) for k in required_keys)
        if not ok:
            st.warning("Einige Pflichtfelder sind noch leer.")
        nxt.button("Next →", on_click=lambda: goto(step + 1))

    # ----------- Summary / Abschluss ----------
    elif step == len(STEPS) + 1:
        st.markdown(
            "<h2 style='text-align:center'>Summary</h2>", unsafe_allow_html=True
        )
        display_summary_overview()
        with st.expander("Alle Daten", expanded=False):
            display_summary()

        ss["data"]["ideal_candidate_profile"] = st.text_area(
            "Ideal Candidate Profile",
            value=ss["data"].get("ideal_candidate_profile", ""),
        )
        ss["data"]["target_industries"] = st.text_area(
            "Target Industries",
            value=ss["data"].get("target_industries", ""),
        )

        st.subheader("Erwartetes Jahresgehalt")
        display_salary_plot()

        st.header("Nächster Schritt – Nutzen Sie die gesammelten Daten!")

        btn_cols = st.columns(6)
        actions = [
            ("Stellenanzeige erstellen", "jobad", generate_jobad),
            ("Guide Vorstellungsgespräch", "interview", generate_interview_sheet),
            (
                "Boolean Search für bessere Search-Engine Findings",
                "boolean",
                generate_boolean_search,
            ),
            ("Arbeitsvertrag erstellen", "contract", generate_contract),
            ("Gehaltsband schätzen", "salary", None),
            ("Gesamtkosten berechnen", "total", None),
        ]

        for col, (label, key, func) in zip(btn_cols, actions):
            with col:
                if st.button(label, key=f"btn_{key}"):
                    if func:
                        with st.spinner("Generiere…"):
                            ss[f"out_{key}"] = asyncio.run(func(ss["data"]))
                    else:
                        if key == "salary":
                            ss[f"out_{key}"] = estimate_salary_range(
                                cast(str, ss["data"].get("job_title", "")),
                                cast(str, ss["data"].get("seniority_level", "")),
                            )
                        elif key == "total":
                            match = re.findall(
                                r"\d+", str(ss["data"].get("salary_range", ""))
                            )
                            if len(match) >= 2:
                                rng = (int(match[0]), int(match[1]))
                            else:
                                rng = (0, int(match[0])) if match else (0, 0)
                            benefits = []
                            if ss["data"].get("health_insurance"):
                                benefits.append("Health Insurance")
                            if ss["data"].get("company_car"):
                                benefits.append("Company Car")
                            if ss["data"].get("flexible_hours"):
                                benefits.append("Flexible Hours")
                            if ss["data"].get("remote_policy"):
                                benefits.append("Home Office Options")
                            if ss["data"].get("learning_budget"):
                                benefits.append("Training Budget")
                            if ss["data"].get("pension_plan"):
                                benefits.append("Pension Plan")
                            ss[f"out_{key}"] = str(
                                calculate_total_compensation(rng, benefits)
                            )

        for label, key, _ in actions:
            if f"out_{key}" in ss:
                ss[f"out_{key}"] = st.text_area(
                    label,
                    value=str(ss.get(f"out_{key}", "")),
                    key=f"txt_{key}",
                    height=200,
                )
                change = st.text_area("Change Request", key=f"chg_{key}", value="")
                if st.button("Apply", key=f"apply_{key}"):
                    if change:
                        ss[f"out_{key}"] = change
                    st.session_state[f"txt_{key}"] = ss[f"out_{key}"]

        step_labels = [title for title, _ in STEPS]
        target = st.selectbox("Zu Schritt springen:", step_labels)
        if st.button("Wechseln"):
            goto(step_labels.index(target) + 1)


if __name__ == "__main__":
    main()
