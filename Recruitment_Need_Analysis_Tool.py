from __future__ import annotations


from streamlit import session_state as ss
import pandas as pd  # type: ignore

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
with open("wizard_schema.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        step = row["step"]
        if step not in SCHEMA:
            SCHEMA[step] = []
        # Parsing options (as list if present)
        options = row["options"].split(";") if row["options"].strip() else None
        row["options"] = options
        SCHEMA[step].append(row)

DATE_KEYS = {"date_of_employment_start", "application_deadline", "probation_period"}

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
    "employment_type",
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
    "DEPARTMENT",
    "ROLE",
    "TASKS",
    "SKILLS",
    "BENEFITS",
    "TARGET_GROUP",
    "INTERVIEW",
    "SUMMARY",
]

STEPS: list[tuple[str, list[str]]] = [
    (name.title().replace("_", " "), [item["key"] for item in SCHEMA[name]])
    for name in ORDER
    if name in SCHEMA
]


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
    "employment_type": _simple("Employment\\s*Type", "Vertragsart", "employment_type"),
    "contract_type": _simple("Contract\\s*Type", "Vertragstyp", "contract_type"),
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
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|to|–)\s*\d{4,6})",
    "salary_range_min": r"(?P<salary_range_min>\d{4,6})\s*(?:-|to|–)\s*\d{4,6}",
    "salary_range_max": r"\d{4,6}\s*(?:-|to|–)\s*(?P<salary_range_max>\d{4,6})",
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
    "line_manager_recv_cv": _simple(
        "Receives\\s*CV", "Erhält\\s*CV", "line_manager_recv_cv"
    ),
    "hr_poc_name": _simple("HR\\s*POC", "Ansprechpartner\\s*HR", "hr_poc_name"),
    "hr_poc_email": r"(?P<hr_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "hr_poc_recv_cv": _simple("Receives\\s*CV", "Erhält\\s*CV", "hr_poc_recv_cv"),
    "finance_poc_name": _simple(
        "Finance\\s*POC", "Ansprechpartner\\s*Finance", "finance_poc_name"
    ),
    "finance_poc_email": r"(?P<finance_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "finance_poc_recv_offer": _simple(
        "Receives\\s*Offer", "Erhält\\s*Angebot", "finance_poc_recv_offer"
    ),
}


LLM_PROMPT = (
    "You parse German or English job ads. "
    "Return ONLY valid JSON where every key maps to an object "
    'with fields "value" (string|null) and "confidence" (0-1).'
)

# Additional lightweight patterns without explicit labels
FALLBACK_PATTERNS: dict[str, str] = {
    "employment_type": r"(?P<employment_type>Vollzeit|Teilzeit|Full[-\s]?time|Part[-\s]?time)",
    "contract_type": r"(?P<contract_type>unbefristet|befristet|permanent|temporary|contract)",
    "seniority_level": r"(?P<seniority_level>Junior|Mid|Senior|Lead|Head|Manager|Einsteiger|Berufserfahren)",
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|bis|to|–)\s*\d{4,6})",
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
    if field_type == "text_area":
        val = st.text_area(label, value=val or "", help=helptext, key=widget_key)

    elif field_type == "selectbox":
        options = meta.get("options", []) or []
        val = st.selectbox(
            label,
            options=options,
            index=options.index(val) if val in options else 0,
            key=widget_key,
            help=helptext,
        )

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


def display_summary() -> None:
    """Show all collected data grouped by step with inline editing."""
    for step_name in ORDER:
        if step_name not in SCHEMA:
            continue
        with st.expander(step_name.title(), expanded=False):
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

    with col3:
        st.markdown("### Skills")
        st.write(f"**Hard Skills:** {val('hard_skills')}")
        st.write(f"**Must Have Skills:** {val('must_have_skills')}")
        st.write(f"**Nice to Have Skills:** {val('nice_to_have_skills')}")
        st.write(f"**Soft Skills:** {val('soft_skills')}")
        st.write(f"**Certifications Required:** {val('certifications_required')}")
        st.write(f"**Domain Expertise:** {val('domain_expertise')}")
        st.write(f"**Language Requirements:** {val('language_requirements')}")

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


def display_interview_section() -> None:
    """Show interview contacts and involvement."""

    def val(field: str) -> str:
        return str(ss["data"].get(field, ""))

    col1, col2, col3 = st.columns(3)

    options = ["Receive CVs", "Receive IV-Invites", "Receive offer"]

    with col1:
        st.markdown("### Need")
        st.write(f"**Line Manager Name:** {val('line_manager_name')}")
        st.write(f"**Line Manager Email:** {val('line_manager_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("line_manager_involve", []),
            key="line_manager_involve",
        )

    with col2:
        st.markdown("### Authority")
        st.write(f"**HR POC Name:** {val('hr_poc_name')}")
        st.write(f"**HR POC Email:** {val('hr_poc_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("hr_poc_involve", []),
            key="hr_poc_involve",
        )

    with col3:
        st.markdown("### Money")
        st.write(f"**Finance POC Name:** {val('finance_poc_name')}")
        st.write(f"**Finance POC Email:** {val('finance_poc_email')}")
        st.multiselect(
            "Involvement",
            options,
            default=ss.get("finance_poc_involve", []),
            key="finance_poc_involve",
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
        "Informationen zum Unternehmen helfen, die Vakanz besser zu verorten und passgenaues Employer Branding zu ermöglichen. "
        "Firmenbezogene Angaben erhöhen die Glaubwürdigkeit und Transparenz gegenüber Kandidat:innen."
    ),
    "DEPARTMENT": (
        "Team- und Abteilungsinfos sind entscheidend, um das Umfeld und die Anforderungen präzise zu erfassen. "
        "So wird klar, wie die Position im Team eingebettet ist und welche Schnittstellen relevant sind."
    ),
    "ROLE": (
        "Die Aufgaben und die Rolle sind das Herzstück der Ausschreibung – hier bitte besonders genau sein. "
        "Je klarer die Rolle beschrieben ist, desto besser passen die späteren Kandidat:innen."
    ),
    "TASKS": (
        "Hier werden alle wesentlichen Aufgaben und Verantwortlichkeiten der Position gesammelt. "
        "Eine transparente Aufgabenbeschreibung hilft Missverständnisse zu vermeiden und Erwartungen zu steuern."
    ),
    "SKILLS": (
        "An dieser Stelle werden die fachlichen und persönlichen Kompetenzen festgehalten, die für die Vakanz wichtig sind. "
        "Eine genaue Definition der Anforderungen erleichtert das Matching im späteren Prozess."
    ),
    "BENEFITS": (
        "In diesem Abschnitt werden die Vorteile und Benefits präsentiert, die das Unternehmen bietet. "
        "Attraktive Zusatzleistungen steigern die Arbeitgeberattraktivität und fördern Bewerbungen."
    ),
    "TARGET_GROUP": (
        "Hier analysierst du die Zielgruppe, für die die Position besonders attraktiv ist. "
        "Durch das Verständnis der Zielgruppe kann die Ansprache und das Sourcing gezielter erfolgen."
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
        st.image(
            "images/color1_logo_transparent_background.png",
            width=200,
        )
        st.markdown(
            """
        <div class="black-text">
            <h2>Recruitment Need Analysis 🧭</h2>
            <p>Welcome! This Tool helps you quickly create a complete vacancy profile.</p>
            <p>Upload a Job Advert. All relevant information will be extracted and preprocessed automatically.</p>
            <p>Afterwards, start discovering missing data in your Specification in order to Minimise Costs and to ensure Maximum Recruitment Success .</p>
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
                "Start discovering missing data in your specification in order to minimise Costs and to ensure maximum recruitment Success"
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
        if st.session_state.job_title and not ss.get("extracted", {}).get("job_title"):
            ss["extracted"]["job_title"] = ExtractResult(
                st.session_state.job_title, 1.0
            )

        if extract_btn and up:
            with st.spinner("Extracting…"):
                file_bytes = up.read()
                text = extract_text_from_file(file_bytes, up.type)
                ss["extracted"] = asyncio.run(extract(text))
                title_res = ss["extracted"].get("job_title")
                if isinstance(title_res, ExtractResult) and title_res.value:
                    ss["extracted"]["job_title"] = title_res
                    ss["data"]["job_title"] = title_res.value
                st.rerun()
    # ----------- 1..n: Wizard -----------
    elif 1 <= step < len(STEPS) + 1:
        step_idx = step - 1
        step_name = ORDER[step_idx]
        meta_fields = SCHEMA[step_name]  # <-- Zuerst setzen!
        fields = [item["key"] for item in meta_fields]
        extr: dict[str, ExtractResult] = ss["extracted"]

        # Headline & Subtitle
        st.markdown(
            f"<h2 style='text-align:center'>{step_name.title()}</h2>",
            unsafe_allow_html=True,
        )
        subtitle = STEP_SUBTITLES.get(step_name, "")
        if subtitle:
            st.markdown(
                f"<div style='text-align:center; color:#bbb; margin-bottom:24px'>{subtitle}</div>",
                unsafe_allow_html=True,
            )

        if step_name == "INTERVIEW":
            display_interview_section()

        # Extrahierte Werte kompakt darstellen
        display_extracted_values_editable(extr, fields, step_name)

        # Prominent fehlende Felder abfragen
        display_missing_inputs(step_name, meta_fields, extr)

        left, right = st.columns(2)
        for meta in meta_fields:
            key = meta["key"]
            result = extr.get(key) if key in extr else ExtractResult()
            is_required = meta.get("is_must", "0") == "1"
            target_col = left if is_required else right
            with target_col:
                show_input(key, result, meta, widget_prefix=step_name)

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

            # Salary prediction chart
            total, parts = predict_annual_salary(
                ss["data"].get("job_title"),
                ss["data"].get("role_description"),
                ss["data"].get("task_list"),
                ss["data"].get("city"),
                hard_sel + soft_sel,
            )
            st.subheader(f"Erwartetes Jahresgehalt: {total} €")
            chart_df = pd.DataFrame(
                {
                    "Component": list(parts.keys()),
                    "Contribution": list(parts.values()),
                }
            ).set_index("Component")
            st.bar_chart(chart_df)

        if step_name == "BENEFITS":
            st.subheader("AI Benefit Suggestions")

            cols = st.columns(3)
            counts = [
                st.number_input(
                    "Count for Job Title",
                    1,
                    50,
                    5,
                    key="count_benefit_title",
                ),
                st.number_input(
                    "Count for Location",
                    1,
                    50,
                    5,
                    key="count_benefit_loc",
                ),
                st.number_input(
                    "Count for Competitors",
                    1,
                    50,
                    5,
                    key="count_benefit_comp",
                ),
            ]

            if cols[0].button("Generate", key="gen_benefit_title"):
                with st.spinner("Generiere…"):
                    try:
                        ss["benefit_suggestions_title"] = asyncio.run(
                            suggest_benefits_by_title(ss["data"], int(counts[0]))
                        )
                    except Exception as e:
                        logging.error("benefit suggestion failed: %s", e)
                        ss["benefit_suggestions_title"] = []

            if cols[1].button("Generate", key="gen_benefit_loc"):
                with st.spinner("Generiere…"):
                    try:
                        ss["benefit_suggestions_location"] = asyncio.run(
                            suggest_benefits_by_location(ss["data"], int(counts[1]))
                        )
                    except Exception as e:
                        logging.error("benefit suggestion failed: %s", e)
                        ss["benefit_suggestions_location"] = []

            if cols[2].button("Generate", key="gen_benefit_comp"):
                with st.spinner("Generiere…"):
                    try:
                        ss["benefit_suggestions_competitors"] = asyncio.run(
                            suggest_benefits_competitors(ss["data"], int(counts[2]))
                        )
                    except Exception as e:
                        logging.error("benefit suggestion failed: %s", e)
                        ss["benefit_suggestions_competitors"] = []

            sel_title = selectable_buttons(
                ss.get("benefit_suggestions_title", []),
                "### Job Title",
                "selected_benefits_title",
                cols=4,
            )

            sel_loc = selectable_buttons(
                ss.get("benefit_suggestions_location", []),
                "### Location",
                "selected_benefits_location",
                cols=4,
            )

            sel_comp = selectable_buttons(
                ss.get("benefit_suggestions_competitors", []),
                "### Competitors",
                "selected_benefits_competitors",
                cols=4,
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

        step_labels = [name.title().replace("_", " ") for name, _ in STEPS]
        target = st.selectbox("Zu Schritt springen:", step_labels)
        if st.button("Wechseln"):
            goto(step_labels.index(target) + 1)


if __name__ == "__main__":
    main()
