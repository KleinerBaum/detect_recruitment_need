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
from typing import cast

from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
import httpx
import streamlit as st
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from dateutil import parser as dateparser
import datetime as dt
import csv
import base64

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
    "Return ONLY valid JSON where every key maps to an object "
    'with fields "value" (string|null) and "confidence" (0-1).'
)


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
def pattern_search(text: str, key: str, pat: str) -> ExtractResult | None:
    """
    Sucht Pattern, säubert gängige Präfixe („Name:“, „City:“ …) und liefert
    ein ExtractResult mit fixer Regex-Confidence 0.9.
    """
    m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
    if not (m and m.group(key)):
        return None

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
    reader = PdfReader(data)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())


@st.cache_data(ttl=24 * 60 * 60)
def docx_text(data: BytesIO) -> str:
    return "\n".join(p.text for p in docx.Document(data).paragraphs)


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
        if (res := pattern_search(text, k, pat))
    }

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
def show_input(key, default, meta):
    field_type = meta.get("field_type", meta.get("field", "text_input"))
    helptext = meta.get("helptext", "")
    required = str(meta.get("is_must", "0")) == "1"
    label = ("★ " if required else "") + meta.get(
        "label", key.replace("_", " ").title()
    )
    # Extract value
    val = getattr(default, "value", default)

    # Field logic
    if field_type == "text_area":
        val = st.text_area(label, value=val or "", help=helptext)

    elif field_type == "selectbox":
        options = meta.get("options", []) or []
        val = st.selectbox(
            label,
            options=options,
            index=options.index(val) if val in options else 0,
            help=helptext,
        )

    elif field_type == "multiselect":
        options = meta.get("options", []) or []
        val = st.multiselect(
            label,
            options=options,
            default=[v for v in (val or []) if v in options],
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

        val = st.number_input(label, value=numeric_val, help=helptext)

    elif field_type == "date_input":
        try:
            date_val = dateparser.parse(str(val)).date() if val else dt.date.today()
        except Exception:
            date_val = dt.date.today()
        val = st.date_input(label, value=date_val, help=helptext)

    elif field_type == "checkbox":
        val = st.checkbox(label, value=str(val).lower() == "true", help=helptext)

    else:
        val = st.text_input(label, value=val or "", help=helptext)

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
    )

    for i, row in edited.iterrows():
        key = rows[cast(int, i)]["_key"]
        ss["data"][key] = row["Wert"]


def display_missing_inputs(meta_fields: list[dict[str, str]], extracted: dict) -> None:
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
        show_input(k, result, meta)


def display_summary() -> None:
    """Show all collected data grouped by step with inline editing."""
    for step_name in ORDER:
        if step_name not in SCHEMA:
            continue
        with st.expander(step_name.title(), expanded=False):
            for meta in SCHEMA[step_name]:
                key = meta["key"]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta)


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
        "Erstelle eine übersichtliche Interviewvorbereitung für Fachbereich und HR. "
        "Stelle Schlüsselkriterien, Muss- und Wunsch-Skills sowie Frageempfehlungen tabellarisch dar. "
        f"Basisdaten: {json.dumps(data, ensure_ascii=False)}"
    )
    chat = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=800,
        messages=[
            {
                "role": "system",
                "content": "Du bist Interviewcoach für HR und Linemanager.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()


# --- c) Boolean Searchstring ---
async def generate_boolean_search(data: dict) -> str:
    """
    Erstellt einen professionellen, auf die Vakanz optimierten Boolean Searchstring für Jobbörsen, LinkedIn, XING etc.
    """
    prompt = (
        "Erstelle einen prägnanten, suchmaschinenoptimierten Boolean Searchstring für Active Sourcing. "
        "Nutze Aufgaben, Anforderungen und Skills als Grundlage. "
        f"Stellenprofil: {json.dumps(data, ensure_ascii=False)}"
    )
    chat = await client.chat.completions.create(
        model="gpt-4o",
        temperature=0.3,
        max_tokens=300,
        messages=[
            {"role": "system", "content": "Du bist Sourcing-Experte."},
            {"role": "user", "content": prompt},
        ],
    )
    return (chat.choices[0].message.content or "").strip()


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


# ── Streamlit main ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Recruitment Need Analysis Tool",
        page_icon="images/color1_logo_transparent_background.png",
        layout="wide",
    )

    ss = st.session_state
    ss.setdefault("step", 0)
    ss.setdefault("data", {})
    ss.setdefault("extracted", {})

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
        # Job title input field + uploader in a single row
        job_title_default = ss["data"].get("job_title")
        if not job_title_default:
            extr_title = ss.get("extracted", {}).get("job_title")
            if isinstance(extr_title, ExtractResult):
                job_title_default = extr_title.value or ""

        if not st.session_state.get("job_title") and job_title_default:
            st.session_state.job_title = job_title_default

        col_left, col_job, col_space, col_upload = st.columns([2, 20, 6, 20])

        with col_job:
            st.text_input(
                "Job Title",
                value=job_title_default or "",
                key="job_title",
            )

        with col_upload:
            up = st.file_uploader(
                "Upload Job Description (PDF or DOCX)", type=["pdf", "docx"]
            )

        ss["data"]["job_title"] = st.session_state.job_title
        if st.session_state.job_title and not ss.get("extracted", {}).get("job_title"):
            ss["extracted"]["job_title"] = ExtractResult(
                st.session_state.job_title, 1.0
            )

        col_space.empty()  # visual spacing between input and upload

        extract_btn = st.button("Extract Vacancy Data", disabled=not up)
        if extract_btn and up:
            with st.spinner("Extracting…"):
                if up.type == "application/pdf":
                    text = pdf_text(BytesIO(up.read()))
                else:
                    text = docx_text(BytesIO(up.read()))
                ss["extracted"] = asyncio.run(extract(text))
                title_res = ss["extracted"].get("job_title")
                if isinstance(title_res, ExtractResult) and title_res.value:
                    ss["extracted"]["job_title"] = title_res
                    ss["data"]["job_title"] = title_res.value
                st.rerun()

        st.button("Next →", on_click=lambda: goto(1))
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

        # Extrahierte Werte kompakt darstellen
        display_extracted_values_editable(extr, fields, step_name)

        # Prominent fehlende Felder abfragen
        display_missing_inputs(meta_fields, extr)

        with st.expander("Alle Felder bearbeiten", expanded=False):
            left, right = st.columns(2)
            for meta in meta_fields:
                key = meta["key"]
                result = extr.get(key) if key in extr else ExtractResult()
                is_required = meta.get("is_must", "0") == "1"
                target_col = left if is_required else right
                with target_col:
                    show_input(key, result, meta)

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

            hard_sel = st.multiselect(
                "AI-Suggested Hard Skills",
                ss.get("hard_skill_suggestions", []),
                default=ss.get("selected_hard_skills", []),
            )
            ss["selected_hard_skills"] = hard_sel
            current_hard = parse_skill_list(ss["data"].get("hard_skills"))
            for sk in hard_sel:
                if sk not in current_hard:
                    current_hard.append(sk)
            ss["data"]["hard_skills"] = ", ".join(current_hard)

            soft_sel = st.multiselect(
                "AI-Suggested Soft Skills",
                ss.get("soft_skill_suggestions", []),
                default=ss.get("selected_soft_skills", []),
            )
            ss["selected_soft_skills"] = soft_sel
            current_soft = parse_skill_list(ss["data"].get("soft_skills"))
            for sk in soft_sel:
                if sk not in current_soft:
                    current_soft.append(sk)
            ss["data"]["soft_skills"] = ", ".join(current_soft)

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
        display_summary()
        st.header("Nächste Schritte – Nutzen Sie die gesammelten Daten!")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            if st.button("Jobad generieren"):
                with st.spinner("Generiere Jobad…"):
                    jobad = asyncio.run(generate_jobad(ss["data"]))
                    st.markdown(jobad)
                    st.info(
                        "Du kannst den Text jetzt anpassen und als PDF exportieren."
                    )
                    if st.button("PDF Download"):
                        download_as_pdf(jobad)
        with col2:
            if st.button("Interviewvorbereitung erstellen"):
                with st.spinner("Generiere Interviewvorbereitung…"):
                    sheet = asyncio.run(generate_interview_sheet(ss["data"]))
                    st.markdown(sheet)
        with col3:
            if st.button("Boolean Searchstring generieren"):
                with st.spinner("Generiere Suchstring…"):
                    boolean_str = asyncio.run(generate_boolean_search(ss["data"]))
                    st.code(boolean_str)
        with col4:
            if st.button("Arbeitsvertrag generieren"):
                with st.spinner("Generiere Vertrag…"):
                    contract = asyncio.run(generate_contract(ss["data"]))
                    st.markdown(contract)

        step_labels = [name.title().replace("_", " ") for name, _ in STEPS]
        target = st.selectbox("Zu Schritt springen:", step_labels)
        if st.button("Wechseln"):
            goto(step_labels.index(target) + 1)


if __name__ == "__main__":
    main()
