from __future__ import annotations

import asyncio, json, re, ast, logging, os
from dataclasses import dataclass
from datetime import datetime
from io import BytesIO
from typing import Iterable
from bs4 import BeautifulSoup
import httpx, streamlit as st
from openai import AsyncOpenAI
from PyPDF2 import PdfReader
import docx  
from dotenv import load_dotenv
from dateutil import parser as dateparser
import datetime as dt
import csv
from collections import defaultdict
import re

SCHEMA = {}
with open("wizard_schema.csv", newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        step = row["step"]
        if step not in SCHEMA:
            SCHEMA[step] = []
        # Parsing options (as list if present)
        options = row["options"].split(";") if row["options"].strip() else None
        row["options"] = options
        SCHEMA[step].append(row)
        
SCHEMA = {}
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

ORDER = ["BASIC", "COMPANY", "DEPARTMENT", "ROLE", "TASKS", "SKILLS",
         "BENEFITS", "TARGET_GROUP", "INTERVIEW", "SUMMARY"]

STEPS: list[tuple[str, list[str]]] = [
    (name.title().replace("_", " "), [item["key"] for item in SCHEMA[name]])
    for name in ORDER if name in SCHEMA
]
st.write("Generated steps:", STEPS)

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
    "seniority_level": _simple("Seniority\\s*Level", "Karrierelevel", "seniority_level"),
    "date_of_employment_start": _simple("Start\\s*Date|Begin\\s*Date", "Eintrittsdatum", "date_of_employment_start"),
    "work_schedule": _simple("Work\\s*Schedule", "Arbeitszeitmodell", "work_schedule"),
    "work_location_city": _simple("City|Ort", "Ort", "work_location_city"),
    # Company core
    "company_name": _simple("Company|Employer", "Unternehmen", "company_name"),
    "city": _simple("City", "Stadt", "city"),
    "company_size": _simple("Company\\s*Size", "Mitarbeiterzahl", "company_size"),
    "industry": _simple("Industry", "Branche", "industry"),
    "headquarters_location": _simple("HQ\\s*Location", "Hauptsitz", "headquarters_location"),
    "place_of_work": _simple("Place\\s*of\\s*Work", "Arbeitsort", "place_of_work"),
    "company_website": r"(?P<company_website>https?://\S+)",
    # Department / team
    "department_name": _simple("Department", "Abteilung", "department_name"),
    "brand_name": _simple("Brand", "", "brand_name"),
    "team_size": _simple("Team\\s*Size", "Teamgröße", "team_size"),
    "team_structure": _simple("Team\\s*Structure", "Teamaufbau", "team_structure"),
    "direct_reports_count": _simple("Direct\\s*Reports", "Direkt\\s*Berichte", "direct_reports_count"),
    "reports_to": _simple("Reports\\s*To", "unterstellt", "reports_to"),
    "supervises": _simple("Supervises", "Führungsverantwortung", "supervises"),
    "tech_stack": _simple("Tech(ology)?\\s*Stack", "Technologien?", "tech_stack"),
    "culture_notes": _simple("Culture", "Kultur", "culture_notes"),
    "team_challenges": _simple("Team\\s*Challenges", "Herausforderungen", "team_challenges"),
    "client_difficulties": _simple("Client\\s*Difficulties", "Kundenprobleme", "client_difficulties"),
    "main_stakeholders": _simple("Stakeholders?", "Hauptansprechpartner", "main_stakeholders"),
    "team_motivation": _simple("Team\\s*Motivation", "Team\\s*Motivationen?", "team_motivation"),
    "recent_team_changes": _simple("Recent\\s*Team\\s*Changes", "Teamveränderungen", "recent_team_changes"),
    "office_language": _simple("Office\\s*Language", "Bürosprache", "office_language"),
    "office_type": _simple("Office\\s*Type", "Bürotyp", "office_type"),
    # Role definition
    "role_description": _simple("Role\\s*Description|Role\\s*Purpose", "Aufgabenstellung", "role_description"),
    "role_type": _simple("Role\\s*Type", "Rollenart", "role_type"),
    "role_keywords": _simple("Role\\s*Keywords?", "Stellenschlüsselwörter", "role_keywords"),
    "role_performance_metrics": _simple("Performance\\s*Metrics", "Rollenkennzahlen", "role_performance_metrics"),
    "role_priority_projects": _simple("Priority\\s*Projects", "Prioritätsprojekte", "role_priority_projects"),
    "primary_responsibilities": _simple("Primary\\s*Responsibilities", "Hauptaufgaben", "primary_responsibilities"),
    "key_deliverables": _simple("Key\\s*Deliverables", "Ergebnisse", "key_deliverables"),
    "success_metrics": _simple("Success\\s*Metrics", "Erfolgskennzahlen", "success_metrics"),
    "main_projects": _simple("Main\\s*Projects", "Hauptprojekte", "main_projects"),
    "travel_required": _simple("Travel\\s*Required", "Reisetätigkeit", "travel_required"),
    "physical_duties": _simple("Physical\\s*Duties", "Körperliche\\s*Arbeit", "physical_duties"),
    "on_call": _simple("On[-\\s]?Call", "Bereitschaft", "on_call"),
    "decision_authority": _simple("Decision\\s*Authority", "Entscheidungsbefugnis", "decision_authority"),
    "process_improvement": _simple("Process\\s*Improvement", "Prozessverbesserung", "process_improvement"),
    "innovation_expected": _simple("Innovation\\s*Expected", "Innovationsgrad", "innovation_expected"),
    "daily_tools": _simple("Daily\\s*Tools", "Tägliche\\s*Tools?", "daily_tools"),
    # Tasks
    "task_list": _simple("Task\\s*List", "Aufgabenliste", "task_list"),
    "key_responsibilities": _simple("Key\\s*Responsibilities", "Hauptverantwortlichkeiten", "key_responsibilities"),
    "technical_tasks": _simple("Technical\\s*Tasks?", "Technische\\s*Aufgaben", "technical_tasks"),
    "managerial_tasks": _simple("Managerial\\s*Tasks?", "Führungsaufgaben", "managerial_tasks"),
    "administrative_tasks": _simple("Administrative\\s*Tasks?", "Verwaltungsaufgaben", "administrative_tasks"),
    "customer_facing_tasks": _simple("Customer[-\\s]?Facing\\s*Tasks?", "Kundenkontaktaufgaben", "customer_facing_tasks"),
    "internal_reporting_tasks": _simple("Internal\\s*Reporting\\s*Tasks", "Berichtsaufgaben", "internal_reporting_tasks"),
    "performance_tasks": _simple("Performance\\s*Tasks", "Leistungsaufgaben", "performance_tasks"),
    "innovation_tasks": _simple("Innovation\\s*Tasks", "Innovationsaufgaben", "innovation_tasks"),
    "task_prioritization": _simple("Task\\s*Prioritization", "Aufgabenpriorisierung", "task_prioritization"),
    # Skills
    "must_have_skills": _simple("Must[-\\s]?Have\\s*Skills?", "Erforderliche\\s*Kenntnisse", "must_have_skills"),
    "nice_to_have_skills": _simple("Nice[-\\s]?to[-\\s]?Have\\s*Skills?", "Wünschenswert", "nice_to_have_skills"),
    "hard_skills": _simple("Hard\\s*Skills", "Fachkenntnisse", "hard_skills"),
    "soft_skills": _simple("Soft\\s*Skills", "Soziale\\s*Kompetenzen?", "soft_skills"),
    "certifications_required": _simple("Certifications?\\s*Required", "Zertifikate", "certifications_required"),
    "language_requirements": _simple("Language\\s*Requirements", "Sprachanforderungen", "language_requirements"),
    "languages_optional": _simple("Languages\\s*Optional", "Weitere\\s*Sprachen", "languages_optional"),
    "analytical_skills": _simple("Analytical\\s*Skills", "Analytische\\s*Fähigkeiten", "analytical_skills"),
    "communication_skills": _simple("Communication\\s*Skills", "Kommunikationsfähigkeiten", "communication_skills"),
    "project_management_skills": _simple("Project\\s*Management\\s*Skills", "Projektmanagementskills?", "project_management_skills"),
    "tool_proficiency": _simple("Tool\\s*Proficiency", "Toolkenntnisse", "tool_proficiency"),
    "tech_stack": _simple("Tech(ology)?\\s*Stack", "Technologien?", "tech_stack"),  # duplicate name OK
    "domain_expertise": _simple("Domain\\s*Expertise", "Fachgebiet", "domain_expertise"),
    "leadership_competencies": _simple("Leadership\\s*Competencies", "Führungskompetenzen?", "leadership_competencies"),
    "industry_experience": _simple("Industry\\s*Experience", "Branchenerfahrung", "industry_experience"),
    "soft_requirement_details": _simple("Soft\\s*Requirement\\s*Details", "Weitere\\s*Anforderungen", "soft_requirement_details"),
    "years_experience_min": _simple("Years\\s*Experience", "Berufserfahrung", "years_experience_min"),
    "it_skills": _simple("IT\\s*Skills", "IT[-\\s]?Kenntnisse", "it_skills"),
    "visa_sponsorship": _simple("Visa\\s*Sponsorship", "Visasponsoring", "visa_sponsorship"),
    # Compensation
    "salary_currency": _simple("Currency", "Währung", "salary_currency"),
    "salary_range": r"(?P<salary_range>\d{4,6}\s*(?:-|to|–)\s*\d{4,6})",
    "salary_range_min": r"(?P<salary_range_min>\d{4,6})\s*(?:-|to|–)\s*\d{4,6}",
    "salary_range_max": r"\d{4,6}\s*(?:-|to|–)\s*(?P<salary_range_max>\d{4,6})",
    "bonus_scheme": _simple("Bonus\\s*Scheme|Bonus\\s*Model", "Bonusregelung", "bonus_scheme"),
    "commission_structure": _simple("Commission\\s*Structure", "Provisionsmodell", "commission_structure"),
    "variable_comp": _simple("Variable\\s*Comp", "Variable\\s*Vergütung", "variable_comp"),
    "vacation_days": _simple("Vacation\\s*Days", "Urlaubstage", "vacation_days"),
    "remote_policy": _simple("Remote\\s*Policy", "Home\\s*Office\\s*Regelung", "remote_policy"),
    "flexible_hours": _simple("Flexible\\s*Hours|Gleitzeit", "Gleitzeit", "flexible_hours"),
    "relocation_support": _simple("Relocation\\s*Support", "Umzugshilfe", "relocation_support"),
    "childcare_support": _simple("Childcare\\s*Support", "Kinderbetreuung", "childcare_support"),
    "learning_budget": _simple("Learning\\s*Budget", "Weiterbildungsbudget", "learning_budget"),
    "company_car": _simple("Company\\s*Car", "Firmenwagen", "company_car"),
    "sabbatical_option": _simple("Sabbatical\\s*Option", "Auszeitmodell", "sabbatical_option"),
    "health_insurance": _simple("Health\\s*Insurance", "Krankenversicherung", "health_insurance"),
    "pension_plan": _simple("Pension\\s*Plan", "Altersvorsorge", "pension_plan"),
    "stock_options": _simple("Stock\\s*Options", "Aktienoptionen", "stock_options"),
    "other_perks": _simple("Other\\s*Perks", "Weitere\\s*Benefits", "other_perks"),
    "pay_frequency": r"(?P<pay_frequency>monthly|annual|yearly|hourly|quarterly)",
    # Recruitment
    "recruitment_contact_email": r"(?P<recruitment_contact_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "recruitment_contact_phone": _simple("Contact\\s*Phone", "Telefon", "recruitment_contact_phone"),
    "recruitment_steps": _simple("Recruitment\\s*Steps", "Bewerbungsprozess", "recruitment_steps"),
    "recruitment_timeline": _simple("Recruitment\\s*Timeline", "Bewerbungszeitplan", "recruitment_timeline"),
    "number_of_interviews": _simple("Number\\s*of\\s*Interviews", "Anzahl\\s*Interviews", "number_of_interviews"),
    "interview_format": _simple("Interview\\s*Format", "Interviewformat", "interview_format"),
    "interview_stage_count": _simple("Interview\\s*Stages?", "Bewerbungsgespräche", "interview_stage_count"),
    "interview_docs_required": _simple("Interview\\s*Docs\\s*Required", "Unterlagen", "interview_docs_required"),
    "assessment_tests": _simple("Assessment\\s*Tests?", "Einstellungstests?", "assessment_tests"),
    "interview_notes": _simple("Interview\\s*Notes", "Interviewnotizen", "interview_notes"),
    "onboarding_process": _simple("Onboarding\\s*Process", "Einarbeitung", "onboarding_process"),
    "onboarding_process_overview": _simple("Onboarding\\s*Overview", "Einarbeitungsüberblick", "onboarding_process_overview"),
    "probation_period": _simple("Probation\\s*Period", "Probezeit", "probation_period"),
    "mentorship_program": _simple("Mentorship\\s*Program", "Mentorenprogramm", "mentorship_program"),
    "welcome_package": _simple("Welcome\\s*Package", "Willkommenspaket", "welcome_package"),
    "application_instructions": _simple("Application\\s*Instructions", "Bewerbungshinweise", "application_instructions"),
    # Key contacts
    "line_manager_name": _simple("Line\\s*Manager", "Fachvorgesetzte?r", "line_manager_name"),
    "line_manager_email": r"(?P<line_manager_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "line_manager_recv_cv": _simple("Receives\\s*CV", "Erhält\\s*CV", "line_manager_recv_cv"),
    "hr_poc_name": _simple("HR\\s*POC", "Ansprechpartner\\s*HR", "hr_poc_name"),
    "hr_poc_email": r"(?P<hr_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "hr_poc_recv_cv": _simple("Receives\\s*CV", "Erhält\\s*CV", "hr_poc_recv_cv"),
    "finance_poc_name": _simple("Finance\\s*POC", "Ansprechpartner\\s*Finance", "finance_poc_name"),
    "finance_poc_email": r"(?P<finance_poc_email>[\w\.-]+@[\w\.-]+\.\w+)",
    "finance_poc_recv_offer": _simple("Receives\\s*Offer", "Erhält\\s*Angebot", "finance_poc_recv_offer"),
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
@st.cache_data(ttl=24*60*60)
def http_text(url: str) -> str:
    html = httpx.get(url, timeout=20).text
    return html_text(html)

@st.cache_data(ttl=24 * 60 * 60)
def pdf_text(data: BytesIO) -> str:
    reader = PdfReader(data)
    return "\n".join(p.extract_text() for p in reader.pages if p.extract_text())

@st.cache_data(ttl=24 * 60 * 60)
def docx_text(data: BytesIO) -> str:
    return "\n".join(p.text for p in docx.Document(data).paragraphs)

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

        raw = safe_json_load(chat.choices[0].message.content)
        for k in subset:
            node = raw.get(k, {})
            val = node.get("value") if isinstance(node, dict) else node
            conf = node.get("confidence", 0.5) if isinstance(node, dict) else 0.5
            out[k] = ExtractResult(val, float(conf) if val else 0.0)
    return out

# ── Extraction orchestrator ---------------------------------------------------
async def extract(text: str) -> dict[str, ExtractResult]:
    interim: dict[str, ExtractResult] = {
        k: res for k, pat in REGEX_PATTERNS.items() if (res := pattern_search(text, k, pat))
    }

    # salary merge
    if (
        "salary_range" not in interim
        and {"salary_range_min", "salary_range_max"} <= interim.keys()
    ):
        interim["salary_range"] = ExtractResult(
            f"{interim['salary_range_min'].value} – {interim['salary_range_max'].value}",
            min(interim["salary_range_min"].confidence, interim["salary_range_max"].confidence),
        )

    missing = [k for k in REGEX_PATTERNS.keys() if k not in interim]
    interim.update(await llm_fill(missing, text))
    return interim

# ── UI helpers ----------------------------------------------------------------
def show_input(key, default, meta):
    field_type = meta.get("field_type", meta.get("field", "text_input"))
    helptext = meta.get("helptext", "")
    required = str(meta.get("is_must", "0")) == "1"
    label = ("★ " if required else "") + meta.get("label", key.replace("_", " ").title())

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


# ── Streamlit main ------------------------------------------------------------
def main():
    st.set_page_config(
        page_title="Recruitment Need Analysis Tool",
        page_icon="🧭",
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
        st.markdown("""
            <h1 style="text-align:center">Recruitment Need Analysis Tool 🧭</h1>
            <div style="text-align:center">
                <p>Welcome! This wizard helps you quickly create a complete vacancy profile.<br>
                Upload a job ad (PDF/DOCX) or paste a URL. All relevant information will be extracted automatically.<br>
                <b>Afterwards, just review and complete missing fields step-by-step.</b></p>
            </div>
        """, unsafe_allow_html=True)

        st.divider()
        up = st.file_uploader("Upload Job Description (PDF or DOCX)", type=["pdf", "docx"])
        url = st.text_input("…or paste a Job Ad URL")

        extract_btn = st.button("Extract Vacancy Data", disabled=not (up or url))
        if extract_btn:
            with st.spinner("Extracting…"):
                if up:
                    if up.type == "application/pdf":
                        text = pdf_text(BytesIO(up.read()))
                    else:
                        text = docx_text(BytesIO(up.read()))
                else:
                    text = http_text(url)
                ss["extracted"] = asyncio.run(extract(text))
            goto(1)
            st.rerun()   # Wichtig: zwingt ein UI-Update nach goto

    # ----------- 1..n: Wizard -----------
    elif 1 <= step < len(STEPS)+1:
        # Classic Wizard Step UI
        step_idx = step - 1
        title, fields = STEPS[step_idx]
        clean_title = title.split("–", 1)[-1].strip()
        data = ss["data"]

        # (Optional: Headline dynamisch anpassen wie gehabt)
        st.header(clean_title)
        extr: dict[str, ExtractResult] = ss["extracted"]

        st.subheader("Auto-extracted values")
        for k in fields:
            res = extr.get(k)
            if res and res.value:
                ss["data"].setdefault(k, res.value)
                st.text(f"{k}: {res.value}  ({res.confidence:.0%})")

        # Inputs in zwei Spalten, Pflichtfelder hervorheben
        step_name = ORDER[step_idx]
        meta_fields = SCHEMA[step_name]
        left, right = st.columns(2)

        for meta in meta_fields:
            key = meta["key"]
            result = extr.get(key) if key in extr else ExtractResult()
            is_required = meta.get("is_must", "0") == "1"
            if is_required:
                with left:
                    show_input(key, result, meta)
            else:
                with right.expander(meta["key"].replace("_", " ").title(), expanded=False):
                    show_input(key, result, meta)

        prev, nxt = st.columns(2)
        prev.button("← Back", disabled=step == 1, on_click=lambda: goto(step - 1))
        required_keys = [meta["key"] for meta in meta_fields if meta.get("is_must", "0") == "1"]
        ok = all(ss["data"].get(k) for k in required_keys)
        nxt.button("Next →", disabled=not ok, on_click=lambda: goto(step + 1))

    # ----------- Summary / Abschluss ----------
    elif step == len(STEPS)+1:
        st.header("Summary")
        st.json(ss["data"], expanded=False)
        st.download_button(
            "Download JSON",
            data=json.dumps(ss["data"], indent=2),
            file_name=f"vacalyser_{datetime.now():%Y%m%d_%H%M}.json",
            mime="application/json",
        )
        st.button("← Edit", on_click=lambda: goto(len(STEPS)))  # zurück zu letztem Step

if __name__ == "__main__":
    main()
