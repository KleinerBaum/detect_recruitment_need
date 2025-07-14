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
from functools import lru_cache
import spacy
from spacy.language import Language
from dataclasses import dataclass
from typing import Any, Literal, Sequence, cast

from io import BytesIO
from pathlib import Path
from bs4 import BeautifulSoup
import httpx
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from openai import AsyncOpenAI

import importlib.util
from dotenv import load_dotenv
from dateutil import parser as dateparser
import datetime as dt
import csv
import base64
import hashlib

_esco_spec = importlib.util.spec_from_file_location(
    "esco_api", Path(__file__).with_name("esco_api.py")
)
assert _esco_spec is not None
_esco_api = importlib.util.module_from_spec(_esco_spec)
assert _esco_spec.loader is not None
_esco_spec.loader.exec_module(_esco_api)
search_occupations = _esco_api.search_occupations
get_skills_for_occupation = _esco_api.get_skills_for_occupation
fetch_occupation_details = _esco_api.fetch_occupation_details
bulk_search_occupations = _esco_api.bulk_search_occupations
get_related_occupations = _esco_api.get_related_occupations
get_skills_for_skill = _esco_api.get_skills_for_skill
get_skill_categories = _esco_api.get_skill_categories
get_occupation_statistics = _esco_api.get_occupation_statistics
suggest = _esco_api.suggest

_spec = importlib.util.spec_from_file_location(
    "file_tools", Path(__file__).with_name("file_tools.py")
)
assert _spec is not None
_file_tools = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_file_tools)
extract_text_from_file = _file_tools.extract_text_from_file
create_pdf = _file_tools.create_pdf
create_docx = _file_tools.create_docx

_vs_spec = importlib.util.spec_from_file_location(
    "vector_search",
    Path(__file__).resolve().parent / "services" / "vector_search.py",
)
assert _vs_spec is not None
_vs_mod = importlib.util.module_from_spec(_vs_spec)
assert _vs_spec.loader is not None
_vs_spec.loader.exec_module(_vs_mod)
VectorStore = _vs_mod.VectorStore

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

DATE_KEYS = {
    "date_of_employment_start",
    "application_deadline",
    "probation_period",
    "contract_end_date",
}
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
    /* red star prefix triggers a red border when the field is empty */
    input.must_req:placeholder-shown {
        border: 1px solid #e74c3c !important;   /* override Streamlit default */
    }
    /* hide default page navigation in sidebar */
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── OpenAI setup ──────────────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("❌ OPENAI_API_KEY missing! Please add it to .env or secrets.toml.")
    st.stop()

client = AsyncOpenAI(api_key=api_key)
vector_store = VectorStore(client)

LOCAL_BENEFITS: dict[str, list[str]] = {
    "düsseldorf": [
        "Fortuna Düsseldorf Vereinsmitgliedschaft",
        "Zugang zur Driving Range auf https://www.golf-duesseldorf.de/driving-range/",
    ],
    "berlin": [
        "BVG-Firmenticket",
        "Urban Sports Club Zuschuss",
    ],
    "münchen": [
        "Vergünstigter Eintritt in die Therme Erding",
        "BahnCard 50 für Pendler",
    ],
    "hamburg": [
        "HVV-ProfiTicket",
        "Segelkurse auf der Alster",
    ],
    "frankfurt": [
        "MuseumsuferCard mit Rabatt",
        "Fitnessstudio-Zuschuss",
    ],
}


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


async def json_chat(
    messages: list[dict[str, str]],
    expected: Sequence[str],
    *,
    max_attempts: int = 2,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_tokens: int = 500,
) -> dict:
    """Return JSON data from the LLM with optional retries."""

    for attempt in range(max_attempts):
        chat = await client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=cast(Any, messages),
            response_format=cast(Any, {"type": "json_object"}),
        )

        raw = chat.choices[0].message.content or ""
        data = safe_json_load(raw)
        if all(k in data for k in expected):
            return data

        missing = [k for k in expected if k not in data]
        logging.warning(
            "LLM output missing keys %s on attempt %s", missing, attempt + 1
        )
        if attempt + 1 < max_attempts:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Your last answer was incomplete. Missing: "
                        f"{', '.join(missing)}. Return JSON with all keys."
                    ),
                }
            )

    return data


def sanitize_value(val: Any) -> str | None:
    """Return a cleaned string or ``None``.

    Strings are stripped of surrounding whitespace and control characters.
    Numeric inputs are formatted without trailing zeros. Everything else is
    coerced to ``str``.
    """

    if val is None:
        return None
    if isinstance(val, (int, float)):
        # format floats without scientific notation
        return ("%g" % val).strip()
    cleaned = str(val)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.strip("\u200b")
    return cleaned or None


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

# Sidebar titles per language
SIDEBAR_TITLES = {
    "English": {
        "BASIC": "Basic",
        "COMPANY": "Company & Department",
        "ROLE": "Role & Tasks",
        "SKILLS": "Skills",
        "BENEFITS": "Benefits",
        "INTERVIEW": "Interview",
        "SUMMARY": "Summary",
    },
    "Deutsch": {
        "BASIC": "Basisdaten",
        "COMPANY": "Unternehmen & Abteilung",
        "ROLE": "Rolle & Aufgaben",
        "SKILLS": "Fähigkeiten",
        "BENEFITS": "Benefits",
        "INTERVIEW": "Interview",
        "SUMMARY": "Zusammenfassung",
    },
}

# Sidebar placeholder shown before any data is entered
SIDEBAR_PLACEHOLDER = {
    "Deutsch": (
        "Hier werden wir alle gesammelten Informationen zu Deiner Vakanz "
        "präsentieren."
    ),
    "English": ("Here we will present all collected information about your vacancy."),
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
    """Return a basic pattern matching a labeled value.

    The pattern captures a single line following either the English or German
    label. It is more restrictive than before to avoid grabbing unrelated text.
    """

    labels = "|".join(part for part in (label_en, label_de) if part)
    label = rf"(?:{labels})"
    bullet = r"^\s*(?:[-*•>]\s*)?"
    return rf"{bullet}{label}\s*[:\-]?\s*(?P<{cap}>[^\n\r]+)"


REGEX_PATTERNS = {
    # BASIC INFO - mandatory
    "job_title": _simple(
        "Job\\s*Title|Position|Jobtitel|Stellentitel|Berufsbezeichnung",
        "",
        "job_title",
    ),
    "employment_type": _simple(
        "Employment\\s*Type|Work\\s*Type",
        (
            "Vertragsart|Beschäftigungsart|Arbeitszeit|Anstellungsart|"
            "Beschäftigungsverhältnis|Art\\s*der\\s*Beschäftigung|Arbeitszeitmodell"
        ),
        "employment_type",
    ),
    "contract_type": _simple(
        "Contract\\s*Type|Type\\s*of\\s*Contract",
        "Vertragstyp|Anstellungsart|Art\\s*des\\s*Vertrags",
        "contract_type",
    ),
    "seniority_level": _simple(
        "Seniority\\s*Level", "Karrierelevel", "seniority_level"
    ),
    "date_of_employment_start": _simple(
        "Start\\s*Date|Begin\\s*Date", "Eintrittsdatum", "date_of_employment_start"
    ),
    "contract_end_date": _simple(
        "End\\s*Date|Contract\\s*End", "Vertragsende|Enddatum", "contract_end_date"
    ),
    "work_schedule": _simple("Work\\s*Schedule", "Arbeitszeitmodell", "work_schedule"),
    "work_location_city": _simple("City|Ort", "Ort", "work_location_city"),
    # Company core
    "company_name": _simple(
        "Company Name|Company|Employer|Firma",
        "Unternehmen|Firmenname",
        "company_name",
    ),
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
    "bonus_percentage": r"(?P<bonus_percentage>\d{1,3}\s*%)",
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
    "You will receive the complete text after HTML parsing. "
    "Return STRICT JSON using the provided keys only. "
    "Every key maps to an object with fields 'value' (string or null) "
    "and 'confidence' (0-1). Missing information must be expressed "
    "as null in the value field."
)

# Additional lightweight patterns without explicit labels
FALLBACK_PATTERNS: dict[str, str] = {
    "employment_type": (
        r"(?P<employment_type>Vollzeit|Teilzeit|Teilzeitkraft|Werkstudent(?:[ei]n)?|"
        r"Praktikum|Mini[-\s]?Job|Freelance|Freelancer|Freiberuflich|"
        r"Internship|Full[-\s]?time|Part[-\s]?time)"
    ),
    "contract_type": (
        r"(?P<contract_type>unbefristet|befristet|befristeter\s*Vertrag|"
        r"festanstellung|permanent|temporary|fixed[-\s]?term|contract|"
        r"freelancer|project|werkvertrag|zeitarbeit|project[-\s]?based)"
    ),
    "seniority_level": r"(?P<seniority_level>Junior|Mid|Senior|Lead|Head|Manager|Einsteiger|Berufserfahren)",
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|bis|to|–)\s*\d{4,6})",
    "work_location_city": r"\bin\s+(?P<work_location_city>[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,}(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß.-]{2,})?)",
}


def search_company_name(text: str) -> ExtractResult | None:
    """Try to guess the company name from common patterns."""

    pat_bei = (
        r"(?:(?<=bei\s)|(?<=at\s))"
        r"(?P<company_name>"
        r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&.,'-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&.,'-]+)*"
        r"(?:\s+(?:GmbH(?: & Co\. KG)?|AG|KG|SE|UG(?: \(haftungsbeschränkt\))?|Inc\.|Ltd\.|LLC|e\.V\.))?"
        r")(?=\s|$)"
    )
    m = re.search(pat_bei, text)
    if m:
        val = m.group("company_name")
        if " " in val or re.search(r"gmbh|ag|kg|se|inc|ltd|llc|e\.v\.", val, re.I):
            return ExtractResult(val, 0.8)

    pat_generic = (
        r"(?P<company_name>"
        r"[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&.,'-]+(?:\s+[A-ZÄÖÜ][A-Za-zÄÖÜäöüß&.,'-]+)*"
        r"(?:\s+(?:GmbH(?: & Co\. KG)?|AG|KG|SE|UG(?: \(haftungsbeschränkt\))?|Inc\.|Ltd\.|LLC|e\.V\.))?"
        r")(?=\s|$)"
    )
    m = re.search(pat_generic, text)
    if m:
        val = m.group("company_name")
        if " " in val or re.search(r"gmbh|ag|kg|se|inc|ltd|llc|e\.v\.", val, re.I):
            return ExtractResult(val, 0.7)
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


@lru_cache(maxsize=1)
def load_ner() -> Language:
    """Load and cache the spaCy NER model."""

    try:
        return spacy.load("xx_ent_wiki_sm")
    except Exception as exc:  # pragma: no cover - log only
        logging.error("spaCy model load failed: %s", exc)
        return spacy.blank("xx")


def ner_fallback(text: str) -> dict[str, ExtractResult]:
    """Return organization and city entities via spaCy."""

    nlp = load_ner()
    if not nlp.pipe_names:  # no NER component available
        return {}
    doc = nlp(text)
    out: dict[str, ExtractResult] = {}
    org = next((e.text for e in doc.ents if e.label_ == "ORG"), None)
    city = next((e.text for e in doc.ents if e.label_ in {"GPE", "LOC"}), None)
    if org:
        out["company_name"] = ExtractResult(org, 0.6)
    if city:
        out["work_location_city"] = ExtractResult(city, 0.6)
    return out


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

    return ExtractResult(value=sanitize_value(val), confidence=0.9)


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


def collect_unique_items(keys: list[str], data: dict[str, Any]) -> list[str]:
    """Return deduplicated items from ``data`` for the given keys."""

    items: list[str] = []
    for key in keys:
        items.extend(parse_skill_list(data.get(key)))
    return sorted({i for i in items if i})


def parse_salary_range(value: str) -> tuple[int, int] | None:
    """Return numeric bounds from a salary range string."""

    digits = [int(x) for x in re.findall(r"\d+", value.replace(",", ""))]
    if len(digits) >= 2:
        return digits[0], digits[1]
    if len(digits) == 1:
        return digits[0], digits[0]
    return None


def salary_deviation_high(
    entered: tuple[int, int], suggested: tuple[int, int], threshold: float = 0.2
) -> bool:
    """Return ``True`` if ``entered`` deviates strongly from ``suggested``."""

    lower, upper = entered
    s_lower, s_upper = suggested
    return (
        abs(lower - s_lower) > s_lower * threshold
        or abs(upper - s_upper) > s_upper * threshold
    )


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


def streamlined_skill_buttons(skills: list[str], label: str, key: str) -> list[str]:
    """Show skill suggestions as styled buttons."""

    return selectable_buttons(skills, label, key, cols=4)


def update_benefit_preferences(city: str) -> None:
    """Update session benefit suggestions based on location."""

    ss["local_benefits"] = LOCAL_BENEFITS.get(city.lower(), [])


def sync_remote_policy(data: dict[str, Any]) -> None:
    """Synchronize remote policy with the selected work schedule."""

    schedule = data.get("work_schedule")
    if schedule == "Remote":
        data["remote_policy"] = "Remote"
        data["remote_percentage"] = 100
    elif schedule == "Hybrid":
        data["remote_policy"] = "Hybrid"
        data.setdefault("remote_percentage", 50)
    else:
        data.setdefault("remote_policy", "Onsite")
        data.pop("remote_percentage", None)


def calc_extraction_progress() -> int:
    """Return extraction completion percentage."""

    total = len(KEY_TO_STEP)
    if not total:
        return 0
    done = 0
    for step in ss.get("extracted", {}).values():
        for res in step.values():
            if getattr(res, "value", None):
                done += 1
    return int(done * 100 / total)


def calc_required_completion() -> int:
    """Return completion percentage for mandatory fields."""

    total = len(MUST_HAVE_KEYS)
    if not total:
        return 0
    done = sum(1 for k in MUST_HAVE_KEYS if not value_missing(k))
    return int(done * 100 / total)


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
    items = [sanitize_value(s) for s in raw.get("skills", []) if s]
    return [i for i in items if i]


async def suggest_hard_skills(data: dict) -> list[str]:
    """Suggest up to 10 relevant hard skills."""
    return await _suggest_skills(data, "hard", 10)


async def suggest_soft_skills(data: dict) -> list[str]:
    """Suggest up to 5 relevant soft skills."""
    return await _suggest_skills(data, "soft", 5)


def get_esco_skills(
    query: str | None = None, *, occupation_uri: str | None = None, limit: int = 20
) -> list[str]:
    """Return ESCO skill titles for a given occupation or search query."""

    if not occupation_uri:
        if not query:
            return []
        occupations = search_occupations(query, limit=1)
        if not occupations:
            return []
        occupation_uri = occupations[0].get("uri", "")

    if not occupation_uri:
        return []

    skills = get_skills_for_occupation(occupation_uri, limit=limit)
    out: list[str] = []
    for item in skills:
        label = item.get("label") or item.get("title")
        if label:
            out.append(str(label))
    return out


async def _suggest_benefits(data: dict, mode: str, count: int) -> list[str]:
    """Return a list of benefits based on the provided mode."""

    job_title = data.get("job_title", "")
    location = data.get("city") or data.get("work_location_city", "")

    loc_key = location.lower()
    if mode == "location":
        local = ss.get("local_benefits") or LOCAL_BENEFITS.get(loc_key)
        if local:
            return cast(list[str], local)[:count]

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
    items = [sanitize_value(b) for b in raw.get("benefits", []) if b]
    return [i for i in items if i]


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
    items = [sanitize_value(x) for x in raw.get(key, []) if x]
    return [i for i in items if i]


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


async def suggest_role_description(data: dict) -> str:
    """Return a short role description based on similar ads."""

    query = f"{data.get('job_title', '')} {data.get('task_list', '')}".strip()
    snippets = await vector_store.search(query, top_k=3) if query else []
    context = "\n".join(snippets)
    prompt = (
        "Write 2–3 sentences describing the role titled "
        f"'{data.get('job_title', '')}'.\nContext:\n{context}"
    )
    return await generate_text(
        prompt,
        model="gpt-4o-mini",
        temperature=0.3,
        max_tokens=150,
        system_msg="You are a professional copywriter for job ads.",
    )


def get_esco_tasks(job_title: str, *, limit: int = 10) -> list[str]:
    """Return up to ``limit`` task titles from ESCO for the given job title."""

    if not job_title:
        return []
    occs = search_occupations(job_title, limit=1)
    if not occs:
        return []
    uri = occs[0].get("uri", "")
    if not uri:
        return []
    skills = get_skills_for_occupation(uri, limit=limit)
    tasks: list[str] = []
    for item in skills:
        label = item.get("label") or item.get("title") or item.get("preferredLabel")
        if label:
            tasks.append(str(label))
    return tasks[:limit]


async def suggest_tasks(data: dict, count: int = 10) -> list[str]:
    """Return task suggestions via OpenAI based on the role data."""

    prompt = (
        f"List up to {count} key tasks for a role titled '{data.get('job_title', '')}'. "
        f"Role description: '{data.get('role_description', '')}'. "
        f"Role type: '{data.get('role_type', '')}'. "
        f"Keywords: '{data.get('role_keywords', '')}'. "
        'Return JSON object {"tasks": [..]} with one task per list item.'
    )
    return await _suggest_items(prompt, "tasks")


async def suggest_recruitment_steps(data: dict, count: int = 5) -> list[str]:
    """Suggest common steps in the recruitment process."""

    prompt = (
        f"List up to {count} typical recruitment steps for a role titled "
        f"'{data.get('job_title', '')}'. "
        'Return JSON object {"items": [..]} with one step per list item.'
    )
    return await _suggest_items(prompt, "items")


async def suggest_interview_questions(data: dict, count: int = 5) -> list[str]:
    """Suggest interview questions tailored to the role."""

    prompt = (
        f"List up to {count} insightful interview questions for a "
        f"{data.get('job_title', '')} position. "
        'Return JSON object {"items": [..]} with one question per list item.'
    )
    return await _suggest_items(prompt, "items")


# ── GPT fill ------------------------------------------------------------------
async def llm_fill(missing_keys: list[str], text: str) -> dict[str, ExtractResult]:
    if not missing_keys:
        return {}

    context_snippets = await vector_store.search(text[:1000], top_k=3)
    context_block = "\n---\n".join(context_snippets)

    CHUNK = 40  # keep replies short
    out: dict[str, ExtractResult] = {}
    for i in range(0, len(missing_keys), CHUNK):
        subset = missing_keys[i : i + CHUNK]
        user_msg = (
            f"Extract the following keys and return STRICT JSON only:\n{subset}\n\n"
            f"CONTEXT:\n{context_block}\n\nTEXT:\n```{text[:12_000]}```"
        )
        raw = await json_chat(
            [
                {"role": "system", "content": LLM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            subset,
        )
        for k in subset:
            node = raw.get(k, {})
            val = node.get("value") if isinstance(node, dict) else node
            conf = node.get("confidence", 0.5) if isinstance(node, dict) else 0.5
            out[k] = ExtractResult(sanitize_value(val), float(conf) if val else 0.0)
    return out


async def llm_validate(data: dict[str, ExtractResult]) -> dict[str, ExtractResult]:
    """Validate extracted values using a language model."""

    if not data:
        return {}

    payload = {k: v.value for k, v in data.items()}
    raw = await json_chat(
        [
            {"role": "system", "content": LLM_PROMPT},
            {
                "role": "user",
                "content": (
                    "Check the following extracted data for plausibility. "
                    "Return corrected JSON with confidence values."
                    f"\nDATA:\n```{json.dumps(payload)}```"
                ),
            },
        ],
        list(data.keys()),
    )
    out = {}
    for k, node in raw.items():
        if isinstance(node, dict):
            val = node.get("value")
            conf = float(node.get("confidence", 0.5)) if val else 0.0
        else:
            val = node
            conf = 0.5 if val else 0.0
        out[k] = ExtractResult(sanitize_value(val), conf)
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

    if {"company_name", "work_location_city"} - interim.keys():
        for k, res in ner_fallback(text).items():
            if k not in interim and res.value:
                interim[k] = res
                if k == "work_location_city" and "city" not in interim:
                    interim["city"] = res

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

    validated = await llm_validate(interim)
    interim.update(validated)

    remaining = [k for k in MUST_HAVE_KEYS if value_missing(k)]
    if remaining:
        interim.update(await llm_fill(remaining, text))
        subset = {k: interim[k] for k in remaining if k in interim}
        if subset:
            interim.update(await llm_validate(subset))
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
    label_text = meta.get("label", key.replace("_", " ").title())
    label = ("★ " if required else "") + label_text
    if required and value_missing(key):
        label = f":red[{label}]"
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
    placeholder = "Required" if required and value_missing(key) else ""
    if field_type == "text_area":
        val = st.text_area(
            label,
            value=val or "",
            help=helptext,
            key=widget_key,
            height=100,
            placeholder=placeholder,
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

    elif field_type == "week_schedule":
        st.markdown(label)
        try:
            schedule = json.loads(val) if val else {}
        except Exception:
            schedule = {}
        result: dict[str, list[str]] = {}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        for day in days:
            times = schedule.get(day, ["08:00", "17:00"])
            try:
                start_default = dt.datetime.strptime(times[0], "%H:%M").time()
            except Exception:
                start_default = dt.time(8, 0)
            try:
                end_default = dt.datetime.strptime(times[1], "%H:%M").time()
            except Exception:
                end_default = dt.time(17, 0)
            cols = st.columns(2)
            with cols[0]:
                start = st.time_input(
                    f"{day} start", value=start_default, key=f"{widget_key}_{day}_s"
                )
            with cols[1]:
                end = st.time_input(
                    f"{day} end", value=end_default, key=f"{widget_key}_{day}_e"
                )
            result[day] = [start.strftime("%H:%M"), end.strftime("%H:%M")]
        val = json.dumps(result)

    elif field_type == "slider":
        digits = [int(x) for x in re.findall(r"\d+", str(val))]
        title = str(ss.get("data", {}).get("job_title", ""))
        level = str(ss.get("data", {}).get("seniority_level", ""))
        suggested: tuple[int, int] | None = None
        if key == "salary_range" and title and level:
            suggested = parse_salary_range(estimate_salary_range(title, level))

        if len(digits) >= 2:
            default_range = (digits[0], digits[1])
        elif len(digits) == 1:
            default_range = (digits[0], digits[0])
        elif suggested:
            default_range = suggested
        else:
            default_range = (50000, 60000)

        selected = st.slider(
            label,
            min_value=0,
            max_value=200000,
            value=default_range,
            step=1000,
            key=widget_key,
            help=helptext,
        )
        val = f"{selected[0]}–{selected[1]}"
        if key == "salary_range" and suggested:
            st.caption(f"Suggested range: {suggested[0]}–{suggested[1]} €")
            if salary_deviation_high(selected, suggested):
                st.warning("Entered salary deviates from suggestion.")

    else:
        val = st.text_input(
            label,
            value=val or "",
            key=widget_key,
            help=helptext,
            placeholder=placeholder,
        )

    # Save to session state
    st.session_state["data"][key] = val
    if key in {"city", "work_location_city"}:
        update_benefit_preferences(str(val))


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
        st.info("No values extracted.")
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

    st.subheader("Missing Data")
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
        "main_stakeholders",
    ]

    with col_a:
        missing_a = [k for k in col_a_keys if k in missing]
        if missing_a:
            st.markdown(f"### Missing Data on {company}")
            for key in missing_a:
                meta = missing[key]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta, widget_prefix="missing_COMPANY")

    with col_b:
        missing_b = [k for k in col_b_keys if k in missing]
        if missing_b:
            st.markdown("### Missing Data on the Department and Team")
            for key in missing_b:
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
    """Show all collected data grouped by step with inline editing.

    Required fields are shown directly, optional fields are displayed collapsed
    within each step.
    """
    for step_name in ORDER:
        if step_name not in SCHEMA:
            continue
        with st.expander(STEP_TITLES.get(step_name, step_name.title()), expanded=False):
            mandatory = [
                meta for meta in SCHEMA[step_name] if meta["key"] in MUST_HAVE_KEYS
            ]
            optional = [
                meta for meta in SCHEMA[step_name] if meta["key"] not in MUST_HAVE_KEYS
            ]

            for meta in mandatory:
                key = meta["key"]
                result = ExtractResult(ss["data"].get(key), 1.0)
                show_input(key, result, meta, widget_prefix=f"summary_{step_name}")

            if optional:
                with st.expander("Additional Fields", expanded=False):
                    for meta in optional:
                        key = meta["key"]
                        result = ExtractResult(ss["data"].get(key), 1.0)
                        show_input(
                            key,
                            result,
                            meta,
                            widget_prefix=f"summary_{step_name}",
                        )


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
        if ss.get("data", {}).get("contract_type") == "Fixed-Term":
            st.write(f"**Contract End Date:** {val('contract_end_date')}")
        st.write(f"**Work Schedule:** {val('work_schedule')}")

    with col2:
        st.markdown("### About the Role")
        st.write(f"**Role Description:** {val('role_description')}")
        st.write(f"**Task List:** {val('task_list')}")
        if ss.get("selected_tasks"):
            st.write(
                f"**Selected Tasks:** {', '.join(cast(list[str], ss['selected_tasks']))}"
            )
        st.write(f"**Technical Tasks:** {val('technical_tasks')}")
        st.write(f"**Managerial Tasks:** {val('managerial_tasks')}")
        st.write(f"**Role Keywords:** {val('role_keywords')}")
        st.write(f"**Ideal Candidate:** {val('ideal_candidate_profile')}")

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
        if ss.get("data", {}).get("bonus_scheme"):
            st.write(f"**Bonus Percentage (%):** {val('bonus_percentage')}")
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
        show_missing("recruitment_contact_email", extr, meta_map, "INTERVIEW")
    with app_cols[1]:
        show_missing("recruitment_contact_phone", extr, meta_map, "INTERVIEW")

    st.subheader("Interview Process")
    proc_cols = st.columns(2)
    with proc_cols[0]:
        show_missing("recruitment_steps", extr, meta_map, "INTERVIEW")
    with proc_cols[1]:
        show_missing("recruitment_timeline", extr, meta_map, "INTERVIEW")

    cols = st.columns(2)
    with cols[0]:
        show_missing("number_of_interviews", extr, meta_map, "INTERVIEW")
    with cols[1]:
        show_missing("interview_format", extr, meta_map, "INTERVIEW")

    st.subheader("Onboarding & Probation")
    onboard_cols = st.columns(3)
    with onboard_cols[0]:
        show_missing("probation_period", extr, meta_map, "INTERVIEW")
    with onboard_cols[1]:
        show_missing("mentorship_program", extr, meta_map, "INTERVIEW")
    with onboard_cols[2]:
        show_missing("welcome_package", extr, meta_map, "INTERVIEW")

    show_missing("onboarding_process", extr, meta_map, "INTERVIEW")
    show_missing("onboarding_process_overview", extr, meta_map, "INTERVIEW")
    show_missing("interview_stage_count", extr, meta_map, "INTERVIEW")
    show_missing("interview_docs_required", extr, meta_map, "INTERVIEW")
    show_missing("assessment_tests", extr, meta_map, "INTERVIEW")
    show_missing("interview_notes", extr, meta_map, "INTERVIEW")
    show_missing("application_instructions", extr, meta_map, "INTERVIEW")


img_path = Path("images/AdobeStock_506577005.jpeg")


# Load image as Base64 so it can be embedded via CSS
def get_base64_image(img_path: Path) -> str:
    """Return the image as a base64 data URL."""

    with open(img_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


# CSS block for a semi-transparent background image
def set_background(image_path: Path, opacity: float = 0.5) -> None:
    """Set a semi-transparent background image via inline CSS."""

    img_url = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(
                rgba(255, 255, 255, {1 - opacity}),
                rgba(255, 255, 255, {1 - opacity})
            ),
            url("{img_url}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# Enable background image
set_background(img_path, opacity=0.5)


def render_header() -> str:
    """Render header with language toggle, logo and page links."""

    labels = {
        "Deutsch": {"adv": "Vorteile", "tech": "Die Technologie dahinter"},
        "English": {"adv": "Advantages", "tech": "The Technology behind"},
    }

    cols = st.columns([2, 2, 2])
    with cols[0]:
        lang = st.radio("", ("Deutsch", "English"), horizontal=True, key="nav_lang")
    with cols[1]:
        st.image("images/color1_logo_transparent_background.png", width=150)
    with cols[2]:
        st.page_link(
            "pages/🏠_Advantages.py",
            label=labels[st.session_state.get("nav_lang", "English")]["adv"],
            icon="🏠",
        )
        st.page_link(
            "pages/💡_Tech_Overview.py",
            label=labels[st.session_state.get("nav_lang", "English")]["tech"],
            icon="💡",
        )
    st.markdown(
        "<h2 style='text-align:center'>Recruitment Need Analysis 🧭</h2>",
        unsafe_allow_html=True,
    )
    return lang


def display_sidebar_data(current_step: int, lang: str) -> None:
    """Show stored values grouped by wizard step in the sidebar.

    Args:
        current_step: Index of the active wizard step.
        lang: Selected UI language ("Deutsch" or "English").
    """

    data = st.session_state.get("data", {})
    extracted = st.session_state.get("extracted", {})
    order = st.session_state.get("ORDER")
    if not order:
        return

    # Show placeholder if no values have been collected yet
    has_content = False
    for step in order:
        values = {k: data.get(k) for k in extracted.get(step, {}) if data.get(k)}
        has_tasks = step == "ROLE" and st.session_state.get("selected_tasks")
        if values or has_tasks:
            has_content = True
            break
    if not has_content:
        st.sidebar.write(SIDEBAR_PLACEHOLDER.get(lang, SIDEBAR_PLACEHOLDER["English"]))
        return

    titles = SIDEBAR_TITLES.get(lang, SIDEBAR_TITLES["English"])
    heading = "Gefundene Daten" if lang == "Deutsch" else "Identified Data"
    heading_shown = False

    for i, step in enumerate(order, start=1):
        values = {k: data.get(k) for k in extracted.get(step, {}) if data.get(k)}
        has_tasks = step == "ROLE" and st.session_state.get("selected_tasks")
        if not values and not has_tasks:
            continue
        if not heading_shown:
            st.sidebar.subheader(heading)
            heading_shown = True
        with st.sidebar.expander(
            titles.get(step, step.title()), expanded=current_step == i
        ):
            for k, v in values.items():
                st.write(f"**{k.replace('_', ' ').title()}:** {v}")
            if has_tasks:
                st.write(
                    "**Selected Tasks:** "
                    + ", ".join(cast(list[str], st.session_state["selected_tasks"]))
                )


def display_extraction_tabs() -> None:
    """Render editable tabs with extracted data."""

    extracted = st.session_state.get("extracted", {})
    order = st.session_state.get("ORDER")
    if not order or not extracted:
        return
    tabs = st.tabs([s.title() for s in order if s in extracted])
    for tab, step in zip(tabs, [s for s in order if s in extracted]):
        keys = [k for k in extracted.get(step, {})]
        with tab:
            display_extracted_values_editable(extracted.get(step, {}), keys, step)


# Mapping for subtitles per wizard step
STEP_SUBTITLES = {
    "BASIC": (
        "This step collects the core vacancy data required for later classification, search and comparison. "
        "The more complete these details are, the easier the role can be found and analysed."
    ),
    "COMPANY": (
        "Information about the company, team and department helps position the vacancy and enables targeted employer branding. "
        "Such details increase credibility and transparency towards candidates."
    ),
    "ROLE": (
        "Here we summarise the role description and tasks. The clearer responsibilities, priorities and tasks are defined, "
        "the better future candidates will fit."
    ),
    "SKILLS": (
        "This section records the technical and personal skills required for the vacancy. "
        "A precise definition of requirements makes matching easier later in the process."
    ),
    "BENEFITS": (
        "This part presents the benefits offered by the company. Attractive perks boost employer appeal and encourage applications."
    ),
    "INTERVIEW": (
        "The interview process and the people involved are documented here. A clear structure ensures a professional candidate experience."
    ),
    "SUMMARY": (
        "In the final step all information is summarised once more. Review the details and export the complete profile."
    ),
}


# AI-Functions


async def generate_jobad(data: dict) -> str:
    """Generate a GDPR-compliant, SEO-optimised job ad from the collected data.

    Returns the job advertisement as Markdown/text.
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
    """Create a compact preparation sheet for line managers and HR.

    Returns a Markdown or HTML table based on key requirements, tasks and desired criteria.
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
    """Create a professional Boolean search string optimised for the vacancy."""
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


async def generate_ideal_candidate_profile(
    data: dict,
    tasks: list[tuple[str, int]],
    skills: list[tuple[str, int]],
) -> str:
    """Create a short profile summary based on prioritized tasks and skills."""

    payload = {
        "tasks": tasks,
        "skills": skills,
    }
    prompt = (
        "Summarise the ideal candidate for this role based on the provided "
        f"job data and priorities.\nJob Data: {json.dumps(data)}\n"
        f"Selected: {json.dumps(payload)}"
    )
    system_msg = "You are an expert recruiter generating concise candidate profiles."
    return await generate_text(
        prompt,
        model="gpt-4o-mini",
        temperature=0.4,
        max_tokens=200,
        system_msg=system_msg,
    )


# --- d) Arbeitsvertrag-Generator ---
async def generate_contract(data: dict) -> str:
    """Create a basic employment contract draft from the extracted core data.

    Note: This is not legal advice and does not replace a lawyer.
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
def scroll_to_top() -> None:
    """Scroll the Streamlit page to the top."""

    st.components.v1.html(
        "<script>window.scrollTo({top: 0, behavior: 'auto'});</script>",
        height=0,
    )


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
    ss.setdefault("selected_tasks", [])
    ss["_used_widget_keys"] = set()
    ss.setdefault("ORDER", ORDER)

    lang_label = render_header()
    display_sidebar_data(ss.get("step", 0), lang_label)

    def goto(i: int):
        ss["step"] = i

    step = ss["step"]
    if ss.get("_prev_step") != step:
        scroll_to_top()
    ss["_prev_step"] = step

    # ----------- 0: Welcome / Upload-Page -----------
    if step == 0:
        intro = {
            "Deutsch": [
                "Willkommen! Dieses Tool hilft dir, schnell ein vollst\u00e4ndiges Anforderungsprofil zu erstellen.",
                "Lade eine Stellenanzeige hoch. Alle relevanten Informationen werden automatisch extrahiert und vorverarbeitet.",
                "Anschlie\u00dfend deckst du fehlende Daten in deiner Spezifikation auf, um Kosten zu minimieren und maximalen Recruiting-Erfolg zu sichern.",
            ],
            "English": [
                "Welcome! This Tool helps you quickly create a complete vacancy profile.",
                "Upload a Job Advert. All relevant information will be extracted and preprocessed automatically.",
                "Afterwards, start discovering missing data in your Specification in order to minimise Costs and to ensure Maximum Recruitment Success.",
            ],
        }

        st.markdown(
            f"<div class='black-text' style='text-align:center;'>"
            f"<p>{intro[lang_label][0]}</p>"
            f"<p>{intro[lang_label][1]}</p>"
            f"<p>{intro[lang_label][2]}</p>"
            "</div>",
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
        left_pad, center_col, right_pad = st.columns([15, 70, 15])

        with center_col:
            status_box = st.empty()
            progress = calc_extraction_progress()
            st.caption(f"Extraktion: {progress}%")
            req = calc_required_completion()
            st.progress(req / 100, text=f"Profile completion: {req}%")
            st.text_input(
                "Stellentitel" if lang_label == "Deutsch" else "Job Title",
                value=job_title_default or "",
                key="job_title",
                label_visibility="visible",
            )

            url = st.text_input(
                "Stellen-URL" if lang_label == "Deutsch" else "Job Ad URL",
                key="job_url",
            )

            if url and ss.get("parsed_url") != url:
                with status_box.container():
                    with st.spinner("Extracting…"):
                        text = http_text(url)
                        if text:
                            flat = asyncio.run(extract(text))
                            ss["extracted"] = group_by_step(flat)
                            title_res = (
                                ss["extracted"].get("BASIC", {}).get("job_title")
                            )
                            if isinstance(title_res, ExtractResult) and title_res.value:
                                ss["data"]["job_title"] = title_res.value
                            ss["parsed_url"] = url
                ss["extraction_success"] = True
                st.rerun()

            up = st.file_uploader(
                (
                    "Stellenbeschreibung hochladen (PDF oder DOCX)"
                    if lang_label == "Deutsch"
                    else "Upload Job Description (PDF or DOCX)"
                ),
                type=["pdf", "docx"],
            )

            if ss.pop("extraction_success", False):
                status_box.success("Extraction complete!", icon="🔥")

            display_extraction_tabs()

            missing_msg = {
                "Deutsch": "Starte danach, fehlende Daten in deiner Spezifikation aufzudecken, um Kosten zu minimieren und maximalen Recruiting-Erfolg zu sichern.",
                "English": "Start discovering missing data in your specification in order to minimise Costs and to ensure maximum recruitment Success",
            }
            st.markdown(
                f"<p style='text-align:center; font-size:calc(1rem + 2pt);'>{missing_msg[lang_label]}</p>",
                unsafe_allow_html=True,
            )

            st.button(
                "Weiter →" if lang_label == "Deutsch" else "Next →",
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

        if up:
            file_bytes = up.getvalue()
            file_hash = hashlib.md5(file_bytes).hexdigest()
            if ss.get("uploaded_file_hash") != file_hash:
                with status_box.container():
                    with st.spinner("Extracting…"):
                        text = extract_text_from_file(file_bytes, up.type)
                        flat = asyncio.run(extract(text))
                        ss["extracted"] = group_by_step(flat)
                        title_res = ss["extracted"].get("BASIC", {}).get("job_title")
                        if isinstance(title_res, ExtractResult) and title_res.value:
                            ss["data"]["job_title"] = title_res.value
                        ss["uploaded_file_hash"] = file_hash
                ss["extraction_success"] = True
                st.rerun()
    # ----------- 1..n: Wizard -----------
    elif 1 <= step < len(STEPS) + 1:
        step_idx = step - 1
        step_name = ORDER[step_idx]
        meta_fields = SCHEMA[step_name]  # <-- Zuerst setzen!
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

        progress = calc_required_completion()
        st.progress(progress / 100, text=f"Profile completion: {progress}%")

        if step_name == "INTERVIEW":
            display_interview_section(meta_fields, extr)

        # Prominently request missing fields
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
                if ss.get("data", {}).get(
                    "contract_type"
                ) == "Fixed-Term" and value_missing("contract_end_date"):
                    show_input(
                        "contract_end_date",
                        extr.get("contract_end_date", ExtractResult()),
                        meta_map["contract_end_date"],
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
                sync_remote_policy(ss["data"])
                schedule = ss.get("data", {}).get("work_schedule")
                if schedule == "Hybrid":
                    default_pct = int(ss.get("data", {}).get("remote_percentage", 50))
                    pct = st.slider(
                        "% Remote",
                        min_value=0,
                        max_value=100,
                        value=default_pct,
                        step=5,
                        key="remote_percentage",
                    )
                    ss["data"]["remote_percentage"] = pct
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
                if value_missing("target_industries"):
                    show_input(
                        "target_industries",
                        extr.get("target_industries", ExtractResult()),
                        meta_map["target_industries"],
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

            with st.expander("Team & Culture Context", expanded=True):
                box_a, box_b, box_c = st.empty(), st.empty(), st.empty()

                def _wrap(container: DeltaGenerator) -> DeltaGenerator:
                    return container.container()

                with _wrap(box_a):
                    st.markdown(
                        "<div style='border:1px solid #ccc;padding:10px'>",
                        unsafe_allow_html=True,
                    )
                    row = st.columns([3, 1])
                    row[0].markdown(f"**{meta_map['tech_stack']['label']}**")
                    if row[1].button("Generate Ideas", key="gen_tech_stack"):
                        with st.spinner("Generating…"):
                            try:
                                ss["tech_stack_suggestions"] = asyncio.run(
                                    suggest_tech_stack(ss["data"])
                                )
                            except Exception as e:
                                logging.error("tech stack suggestion failed: %s", e)
                                ss["tech_stack_suggestions"] = []
                    show_missing("tech_stack", extr, meta_map, step_name)
                    sel_ts = st.pills(
                        "",
                        ss.get("tech_stack_suggestions", []),
                        selection_mode="multi",
                        key="sel_tech_stack",
                    )
                    current_ts = parse_skill_list(ss["data"].get("tech_stack"))
                    for s in sel_ts or []:
                        if s not in current_ts:
                            current_ts.append(s)
                    ss["data"]["tech_stack"] = ", ".join(current_ts)
                    st.markdown("</div>", unsafe_allow_html=True)

                with _wrap(box_b):
                    st.markdown(
                        "<div style='border:1px solid #ccc;padding:10px'>",
                        unsafe_allow_html=True,
                    )
                    row = st.columns([3, 1])
                    row[0].markdown(f"**{meta_map['team_challenges']['label']}**")
                    if row[1].button("Generate Ideas", key="gen_team_challenges"):
                        with st.spinner("Generating…"):
                            try:
                                ss["team_challenges_suggestions"] = asyncio.run(
                                    suggest_team_challenges(ss["data"])
                                )
                            except Exception as e:
                                logging.error("team challenge suggestion failed: %s", e)
                                ss["team_challenges_suggestions"] = []
                    show_missing("team_challenges", extr, meta_map, step_name)
                    sel_tc = st.pills(
                        "",
                        ss.get("team_challenges_suggestions", []),
                        selection_mode="multi",
                        key="sel_team_challenges",
                    )
                    cur_tc = parse_skill_list(ss["data"].get("team_challenges"))
                    for s in sel_tc or []:
                        if s not in cur_tc:
                            cur_tc.append(s)
                    ss["data"]["team_challenges"] = ", ".join(cur_tc)
                    row = st.columns([3, 1])
                    row[0].markdown(f"**{meta_map['client_difficulties']['label']}**")
                    if row[1].button("Generate Ideas", key="gen_client_difficulties"):
                        with st.spinner("Generating…"):
                            try:
                                ss["client_difficulties_suggestions"] = asyncio.run(
                                    suggest_client_difficulties(ss["data"])
                                )
                            except Exception as e:
                                logging.error(
                                    "client difficulty suggestion failed: %s", e
                                )
                                ss["client_difficulties_suggestions"] = []
                    show_missing("client_difficulties", extr, meta_map, step_name)
                    sel_cd = st.pills(
                        "",
                        ss.get("client_difficulties_suggestions", []),
                        selection_mode="multi",
                        key="sel_client_difficulties",
                    )
                    cur_cd = parse_skill_list(ss["data"].get("client_difficulties"))
                    for s in sel_cd or []:
                        if s not in cur_cd:
                            cur_cd.append(s)
                    ss["data"]["client_difficulties"] = ", ".join(cur_cd)
                    st.markdown("</div>", unsafe_allow_html=True)

                with _wrap(box_c):
                    st.markdown(
                        "<div style='border:1px solid #ccc;padding:10px'>",
                        unsafe_allow_html=True,
                    )
                    row = st.columns([3, 1])
                    row[0].markdown(f"**{meta_map['recent_team_changes']['label']}**")
                    if row[1].button("Generate Ideas", key="gen_recent_team_changes"):
                        with st.spinner("Generating…"):
                            try:
                                ss["recent_team_changes_suggestions"] = asyncio.run(
                                    suggest_recent_team_changes(ss["data"])
                                )
                            except Exception as e:
                                logging.error("recent changes suggestion failed: %s", e)
                                ss["recent_team_changes_suggestions"] = []
                    show_missing("recent_team_changes", extr, meta_map, step_name)
                    sel_rc = st.pills(
                        "",
                        ss.get("recent_team_changes_suggestions", []),
                        selection_mode="multi",
                        key="sel_recent_team_changes",
                    )
                    cur_rc = parse_skill_list(ss["data"].get("recent_team_changes"))
                    for s in sel_rc or []:
                        if s not in cur_rc:
                            cur_rc.append(s)
                    ss["data"]["recent_team_changes"] = ", ".join(cur_rc)
                    if value_missing("culture_notes"):
                        show_input(
                            "culture_notes",
                            extr.get("culture_notes", ExtractResult()),
                            meta_map["culture_notes"],
                            widget_prefix=step_name,
                        )
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
                    st.markdown("</div>", unsafe_allow_html=True)

            company_keys = [k for k, step in KEY_TO_STEP.items() if step == "COMPANY"]
            if all(not value_missing(k) for k in company_keys):
                if not ss.get("company_step_done"):
                    st.toast(
                        "🎉 Congratulations! All Company & Department data entered."
                    )
                    ss["company_step_done"] = True
            else:
                ss["company_step_done"] = False

        elif step_name == "ROLE":
            meta_map = {m["key"]: m for m in meta_fields}

            st.subheader("Role Summary")
            if st.button("Generate Role Description", key="gen_role_desc"):
                with st.spinner("Generating …"):
                    try:
                        ss["data"]["role_description"] = asyncio.run(
                            suggest_role_description(ss["data"])
                        )
                    except Exception as e:  # pragma: no cover - log only
                        logging.error("role description generation failed: %s", e)
                        ss["data"]["role_description"] = ""

            role_desc = ss.get("data", {}).get("role_description")
            if role_desc:
                st.text_area(
                    meta_map["role_description"]["label"],
                    value=role_desc,
                    key=f"{step_name}_role_description",
                    disabled=True,
                )
            else:
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
                if ss.get("data", {}).get("supervises") and value_missing(
                    "direct_reports_count"
                ):
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
                elif not ss.get("data", {}).get("supervises"):
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
            row = st.columns([2, 1, 1])
            if row[1].button("AI Tasks", key="gen_ai_tasks"):
                with st.spinner("Generating…"):
                    try:
                        ss["ai_task_suggestions"] = asyncio.run(
                            suggest_tasks(ss["data"])
                        )
                    except Exception as e:
                        logging.error("task suggestion failed: %s", e)
                        ss["ai_task_suggestions"] = []
            if row[2].button("ESCO Tasks", key="gen_esco_tasks"):
                with st.spinner("Fetching…"):
                    try:
                        title = ss.get("data", {}).get("job_title", "")
                        ss["esco_task_suggestions"] = get_esco_tasks(title)
                    except Exception as e:
                        logging.error("ESCO task lookup failed: %s", e)
                        ss["esco_task_suggestions"] = []
            ai_sel = st.pills(
                "",
                ss.get("ai_task_suggestions", []),
                selection_mode="multi",
                key="sel_ai_tasks",
            )
            esco_sel = st.pills(
                "",
                ss.get("esco_task_suggestions", []),
                selection_mode="multi",
                key="sel_esco_tasks",
            )
            chosen_tasks = cast(list[str], ss.setdefault("selected_tasks", []))
            for t in (ai_sel or []) + (esco_sel or []):
                if t not in chosen_tasks:
                    chosen_tasks.append(t)
            ss["selected_tasks"] = chosen_tasks
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
                if ss.get("data", {}).get("on_call"):
                    show_input(
                        "on_call_expectations",
                        extr.get("on_call_expectations", ExtractResult()),
                        meta_map["on_call_expectations"],
                        widget_prefix=step_name,
                    )
                else:
                    ss["data"]["on_call_expectations"] = ""
            with cols[2]:
                show_missing("physical_duties", extr, meta_map, step_name)
                if ss.get("data", {}).get("physical_duties"):
                    show_input(
                        "physical_duties_description",
                        extr.get("physical_duties_description", ExtractResult()),
                        meta_map["physical_duties_description"],
                        widget_prefix=step_name,
                    )
                else:
                    ss["data"]["physical_duties_description"] = ""
            cols = st.columns(3)
            with cols[0]:
                show_missing("travel_required", extr, meta_map, step_name)
                travel_val = ss.get("data", {}).get("travel_required")
                if travel_val and str(travel_val) != "No":
                    show_input(
                        "travel_region",
                        extr.get("travel_region", ExtractResult()),
                        meta_map["travel_region"],
                        widget_prefix=step_name,
                    )
                    show_input(
                        "travel_length_days",
                        extr.get("travel_length_days", ExtractResult()),
                        meta_map["travel_length_days"],
                        widget_prefix=step_name,
                    )
                    show_input(
                        "travel_frequency_number",
                        extr.get("travel_frequency_number", ExtractResult()),
                        meta_map["travel_frequency_number"],
                        widget_prefix=step_name,
                    )
                    show_input(
                        "travel_frequency_unit",
                        extr.get("travel_frequency_unit", ExtractResult()),
                        meta_map["travel_frequency_unit"],
                        widget_prefix=step_name,
                    )
                    show_input(
                        "weekend_travel",
                        extr.get("weekend_travel", ExtractResult()),
                        meta_map["weekend_travel"],
                        widget_prefix=step_name,
                    )
                    show_input(
                        "travel_details",
                        extr.get("travel_details", ExtractResult()),
                        meta_map["travel_details"],
                        widget_prefix=step_name,
                    )
                else:
                    for k in [
                        "travel_region",
                        "travel_length_days",
                        "travel_frequency_number",
                        "travel_frequency_unit",
                        "weekend_travel",
                        "travel_details",
                    ]:
                        ss["data"][k] = ""

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
                with st.popover("Select industry"):
                    ind_val = st.radio(
                        "Industry",
                        INDUSTRY_OPTIONS,
                        index=(
                            INDUSTRY_OPTIONS.index(
                                ss.get("data", {}).get("industry_experience", [""])[0]
                            )
                            if ss.get("data", {}).get("industry_experience")
                            else 0
                        ),
                        key=f"{step_name}_industry_list",
                    )
                ss["data"]["industry_experience"] = [ind_val]
            else:
                ss["data"]["industry_experience"] = []

            dom_required = st.checkbox(
                "Domain expertise",
                value=bool(ss.get("data", {}).get("domain_expertise_required")),
                key=f"{step_name}_domain_exp_req",
            )
            ss["data"]["domain_expertise_required"] = dom_required
            if dom_required:
                default_domains = cast(
                    list[str], ss.get("data", {}).get("domain_expertise", [])
                )
                with st.popover("Select domains"):
                    doms = st.multiselect(
                        "Domains",
                        DOMAIN_OPTIONS,
                        default=default_domains,
                        key=f"{step_name}_domain_list",
                    )
                ss["data"]["domain_expertise"] = doms
            else:
                ss["data"]["domain_expertise"] = []

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
                if value_missing("years_experience_min"):
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
                    if ss.get("data", {}).get("remote_policy") not in {
                        "",
                        "Onsite",
                    } and value_missing("remote_percentage"):
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
                        pct = st.number_input(
                            meta_map["bonus_percentage"]["label"],
                            min_value=0.0,
                            max_value=100.0,
                            step=0.5,
                            key="bonus_percentage",
                        )
                        ss["data"]["bonus_percentage"] = pct

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
                meta_map = {m["key"]: m for m in meta_fields}
                current_cols = 2
                cols = st.columns(current_cols)
                col_idx = 0

                for meta in meta_fields:
                    key = meta["key"]
                    field_type = meta.get("field_type", meta.get("field", "text_input"))

                    if field_type == "text_area":
                        cols = st.columns(1)
                        with cols[0]:
                            show_missing(key, extr, meta_map, step_name)
                        cols = st.columns(current_cols)
                        col_idx = 0
                        continue

                    needed = 3 if field_type == "checkbox" else 2
                    if needed != current_cols or col_idx >= needed:
                        cols = st.columns(needed)
                        current_cols = needed
                        col_idx = 0

                    with cols[col_idx]:
                        show_missing(key, extr, meta_map, step_name)

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
            soft_sel = streamlined_skill_buttons(
                ss.get("soft_skill_suggestions", []),
                soft_label,
                "selected_soft_skills",
            )
            current_soft = parse_skill_list(ss["data"].get("soft_skills"))
            for sk in soft_sel:
                if sk not in current_soft:
                    current_soft.append(sk)
            ss["data"]["soft_skills"] = ", ".join(current_soft)

            st.subheader("ESCO Occupation Lookup")
            query = st.text_input("Search Occupation", key="esco_query")
            if st.button("Search", key="esco_search") and query:
                ss["esco_results"] = search_occupations(query)
                ss.pop("esco_skill_suggestions", None)

            bulk = st.text_area("Bulk Titles", key="esco_bulk")
            if st.button("Bulk Search", key="esco_bulk_btn") and bulk.strip():
                titles = [t.strip() for t in bulk.splitlines() if t.strip()]
                ss["bulk_esco"] = bulk_search_occupations(titles)

            if "bulk_esco" in ss:
                for t, res in cast(dict[str, list[dict]], ss["bulk_esco"]).items():
                    title = res[0].get("title") if res else "-"
                    st.caption(f"{t} → {title}")

            occs = cast(list[dict[str, Any]], ss.get("esco_results", []))
            if occs:
                labels = [o.get("label") or o.get("title", "") for o in occs]
                idx = st.selectbox(
                    "Select Occupation",
                    range(len(labels)),
                    format_func=lambda i: labels[i],
                    key="esco_select",
                )
                occ_uri = occs[idx].get("uri", "")
                if occ_uri:
                    details = fetch_occupation_details(occ_uri)
                    if desc := details.get("description"):
                        st.info(desc)
                    alt = details.get("altLabels") or []
                    if alt:
                        st.caption("Also known as: " + ", ".join(cast(list[str], alt)))
                    stats = get_occupation_statistics(occ_uri)
                    st.caption(
                        f"Languages: {stats.get('languages', 0)} – Related skills: {stats.get('skills', 0)}"
                    )
                    if st.button("Related Occupations", key="esco_related"):
                        ss["related_occs"] = get_related_occupations(occ_uri)
                    if st.button("Fetch Skills", key="esco_fetch_skills"):
                        ss["esco_skill_suggestions"] = get_esco_skills(
                            occupation_uri=occ_uri
                        )

            if ss.get("related_occs"):
                rel_labels = [
                    o.get("title", "") for o in cast(list[dict], ss["related_occs"])
                ]
                st.write("Related:", ", ".join(rel_labels))

            if ss.get("esco_skill_suggestions"):
                esco_sel = streamlined_skill_buttons(
                    cast(list[str], ss["esco_skill_suggestions"]),
                    "ESCO Skills",
                    "selected_esco_skills",
                )
                for sk in esco_sel:
                    if sk not in current_hard:
                        current_hard.append(sk)
                ss["data"]["hard_skills"] = ", ".join(current_hard)

                if esco_sel:
                    chosen = esco_sel[-1]
                    sugg = suggest(chosen, type_="skill", limit=1)
                    if sugg:
                        s_uri = sugg[0].get("uri", "")
                        cats = get_skill_categories(s_uri)
                        subs = get_skills_for_skill(s_uri)
                        if cats:
                            st.caption("Categories: " + ", ".join(cats))
                        sub_names = [s.get("title", "") for s in subs][:5]
                        if sub_names:
                            st.caption("Related Skills: " + ", ".join(sub_names))

            with st.container(border=True):
                st.subheader("Ideal Candidate Profile")
                st.write(f"**Job Title:** {ss['data'].get('job_title', '')}")
                st.write(
                    f"**Work Location City:** {ss['data'].get('work_location_city', '')}"
                )
                st.write(f"**Contract Type:** {ss['data'].get('contract_type', '')}")

                task_keys = [
                    "task_list",
                    "technical_tasks",
                    "managerial_tasks",
                    "administrative_tasks",
                    "customer_facing_tasks",
                    "internal_reporting_tasks",
                    "performance_tasks",
                    "innovation_tasks",
                ]
                tasks_all = collect_unique_items(task_keys, ss.get("data", {}))
                st.markdown("#### Tasks")
                selected_tasks: list[tuple[str, int]] = []
                for i, task in enumerate(tasks_all):
                    cols = st.columns([0.7, 0.3])
                    chk = cols[0].checkbox(task, key=f"chk_task_{i}")
                    yrs = cols[1].number_input(
                        "Years",
                        min_value=0,
                        step=1,
                        key=f"yrs_task_{i}",
                    )
                    if chk:
                        selected_tasks.append((task, int(yrs)))

                skill_keys = [
                    "must_have_skills",
                    "nice_to_have_skills",
                    "certifications_required",
                    "language_requirements",
                    "tool_proficiency",
                    "tech_stack",
                    "it_skills",
                ]
                skills_all = collect_unique_items(skill_keys, ss.get("data", {}))
                st.markdown("#### Skills")
                selected_skills: list[tuple[str, int]] = []
                for i, sk in enumerate(skills_all):
                    cols = st.columns([0.7, 0.3])
                    chk = cols[0].checkbox(sk, key=f"chk_skill_{i}")
                    yrs = cols[1].number_input(
                        "Years",
                        min_value=0,
                        step=1,
                        key=f"yrs_skill_{i}",
                    )
                    if chk:
                        selected_skills.append((sk, int(yrs)))

                if st.button("Does your ideal Profile look like this?", key="gen_icp"):
                    with st.spinner("Generating…"):
                        try:
                            ss["data"]["ideal_candidate_profile"] = asyncio.run(
                                generate_ideal_candidate_profile(
                                    ss.get("data", {}), selected_tasks, selected_skills
                                )
                            )
                            sal, parts = predict_annual_salary(
                                cast(str | None, ss["data"].get("job_title")),
                                cast(str | None, ss["data"].get("role_description")),
                                " ".join([t for t, _ in selected_tasks]),
                                cast(str | None, ss["data"].get("city")),
                                [s for s, _ in selected_skills],
                            )
                            ss["data"]["salary_range"] = f"{sal - 2000}–{sal + 2000} €"
                            ss["salary_breakdown"] = [
                                "Base salary: 30000 €",
                                f"Job title impact: {parts['job_title']} €",
                                f"Role description impact: {parts['role_description']} €",
                                f"Tasks impact: {parts['task_list']} €",
                                f"Location impact: {parts['location']} €",
                                f"Skills impact: {parts['skills']} €",
                            ]
                        except Exception as e:
                            logging.error("ideal profile generation failed: %s", e)
                if ss.get("data", {}).get("ideal_candidate_profile"):
                    st.markdown(ss["data"]["ideal_candidate_profile"])
                    st.markdown(
                        f"**Gehalts-Estimate:** {ss['data'].get('salary_range', '')}"
                    )
                    for p in ss.get("salary_breakdown", []):
                        st.markdown(f"- {p}")

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
                        with st.spinner("Generating…"):
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
            st.warning("Some required fields are still empty.")
        nxt.button("Next →", on_click=lambda: goto(step + 1))

    # ----------- Summary / Abschluss ----------
    elif step == len(STEPS) + 1:
        st.markdown(
            "<h2 style='text-align:center'>Summary</h2>", unsafe_allow_html=True
        )
        display_summary_overview()
        with st.expander("All Data", expanded=False):
            display_summary()

        # Ideal Candidate Profile is now collected in the BASIC step

        st.subheader("Expected Annual Salary")
        display_salary_plot()

        ss.setdefault("font_choice", "Arial")
        st.selectbox(
            "Font",
            ["Arial", "Helvetica", "Courier", "Times"],
            key="font_choice",
        )
        logo_file = st.file_uploader(
            "Upload Logo", type=["png", "jpg", "jpeg"], key="logo_file"
        )
        logo_bytes = logo_file.getvalue() if logo_file else None

        st.header("Next Step – Use the collected data!")

        btn_cols = st.columns(6)
        actions = [
            ("Create Job Ad", "jobad", generate_jobad),
            ("Interview Guide", "interview", generate_interview_sheet),
            (
                "Boolean Search for better search engine findings",
                "boolean",
                generate_boolean_search,
            ),
            ("Create Contract", "contract", generate_contract),
            ("Estimate Salary Range", "salary", None),
            ("Calculate Total Cost", "total", None),
        ]

        for col, (label, key, func) in zip(btn_cols, actions):
            with col:
                if st.button(label, key=f"btn_{key}"):
                    if func:
                        with st.spinner("Generating…"):
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

        def apply_change(k: str) -> None:
            """Update generated text with optional user changes."""
            change_val = st.session_state.get(f"chg_{k}", "")
            if change_val:
                ss[f"out_{k}"] = change_val
            st.session_state[f"txt_{k}"] = ss[f"out_{k}"]

        for label, key, _ in actions:
            if f"out_{key}" in ss:
                ss[f"out_{key}"] = st.text_area(
                    label,
                    value=str(ss.get(f"out_{key}", "")),
                    key=f"txt_{key}",
                    height=200,
                )
                st.text_area("Change Request", key=f"chg_{key}", value="")
                st.button(
                    "Apply", key=f"apply_{key}", on_click=apply_change, args=(key,)
                )
                pdf_bytes = create_pdf(
                    ss[f"out_{key}"],
                    font=ss.get("font_choice", "Arial"),
                    logo=logo_bytes,
                )
                st.download_button(
                    "Download PDF",
                    pdf_bytes,
                    file_name=f"{key}.pdf",
                    mime="application/pdf",
                )
                doc_bytes = create_docx(
                    ss[f"out_{key}"],
                    font=ss.get("font_choice", "Arial"),
                    logo=logo_bytes,
                )
                st.download_button(
                    "Download DOCX",
                    doc_bytes,
                    file_name=f"{key}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )

        step_labels = [title for title, _ in STEPS]
        target = st.selectbox("Jump to step:", step_labels)
        if st.button("Switch"):
            goto(step_labels.index(target) + 1)


if __name__ == "__main__":
    main()
