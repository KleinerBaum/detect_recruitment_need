# pages/advantages.py
"""Streamlit-Seite: Vorteile / Advantages von Vacalyser

Bietet einen Sprachumschalter (Deutsch ↔ English) und vier Tabs für die
Zielgruppen Line Manager, Recruiter, Unternehmen und Bewerber.
Die Bullet‑Listen lassen sich jederzeit kürzen oder erweitern, indem du die
entsprechenden Listen unten anpasst.
"""

import streamlit as st
from typing import List, Dict

# ---------------------------------------------------------------------------
# Page‑Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Vacalyser – Vorteile / Advantages",
    page_icon="💡",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sprachumschalter
# ---------------------------------------------------------------------------
lang: str = st.radio("🌐 Sprache / Language", ("Deutsch", "English"), horizontal=True)

# ---------------------------------------------------------------------------
# Daten: Benefits in DE & EN
# ---------------------------------------------------------------------------
BenefitsDict = Dict[str, Dict[str, List[str]]]
benefits: BenefitsDict = {
    "Deutsch": {
        "Line Manager": [
            "Sofortiger Kompetenz‑Abgleich zwischen gewünschtem Skill‑Set und Marktverfügbarkeit",
            "Live‑Budget‑Kalkulation auf Basis aktueller Gehaltsbenchmarks",
            "Ampel‑Warnsystem für unrealistische Anforderungskombinationen",
            "Automatische Priorisierung kritischer Must‑have‑Skills",
            "Zeitersparnis durch One‑Pager‑Summary anstelle langer E‑Mail‑Threads",
            'Interaktive Szenario‑Planung ("Was kostet es, wenn wir auf Senior‑ statt Mid‑Level gehen?")',
            "Rollen‑Klarheit dank vordefinierter Verantwortlichkeitsmatrix (RACI)",
            "Benutzerdefinierte KPI‑Dashboards (Time‑to‑Hire, Quality‑of‑Hire, Cost‑per‑Hire)",
            'Versionshistorie für spätere Audits ("wer hat wann was geändert?")',
            "Automatisierte Reminder bei überfälligen Feedback‑Schritten",
            "KI‑gestützte Formulierungshilfen für attraktive Benefit‑Formulierungen",
            "Skill‑Gap‑Heatmap für internes Upskilling vs. externes Hiring",
            "Instant‑Marktreport (Kandidatendichte, Konkurrenzdruck, Standortattraktivität)",
            "Genehmigungs‑Workflow mit elektronischer Signatur",
            "Konfliktfreies Headcount‑Tracking über mehrere Cost‑Center hinweg",
            "Self‑Service‑Reports ohne BI‑Abhängigkeit",
            "Compliance‑Check (AGG, Equal Pay, EU‑AI‑Act‑Konformität)",
            "Benchmarks gegen Unternehmensrichtlinien (Salary Bands, Remote‑Policy)",
            "Mehrsprachige Oberfläche für globale Teams",
            "Export‑Funktion zu PowerPoint, PDF, Excel",
            "Integrations‑Hooks (Slack, Teams, Jira) für nahtlose Kollaboration",
            '"Next‑Best‑Action"‑Hints (z. B. "Interview‑Panel einladen")',
            "Kalender‑Sync zur Blockierung kritischer Deadlines",
            "Realtime‑Push‑Notifications bei Marktveränderungen (Tarifabschlüsse etc.)",
            "Durchgängige Barrierefreiheit (WCAG 2.2)",
            "Dark‑Mode für lange Arbeitssitzungen",
            "Mobile‑Responsive für Genehmigungen on the go",
            "Data‑Privacy‑Shield trennt sensible Kandidatendaten von Management‑Info",
            "On‑Demand‑Training‑Snippets (Embedded Videos, Tooltips)",
            "Gamifizierte Completion‑Tracker zur Motivation",
        ],
        "Recruiter": [
            "Automatisches Parsing von JD‑Entwürfen, CVs & Links",
            "Datenbank‑Vernetzung (ATS, CRM, LinkedIn Recruiter, GitHub)",
            "KPIs in Echtzeit (Sourcing‑Conversion‑Rate, Pipeline‑Velocity)",
            'KI‑basierte Suchstrings generieren ("Boolean‑Builder 2.0")',
            "Kandidaten‑Personas aus Text‑ und Profilanalyse",
            "Bulk‑Mail‑Personalisierung mit A/B‑Subject‑Lines",
            "Job‑Ad‑SEO‑Check (Google for Jobs, Indeed)",
            "Bias‑Detection & Inclusive‑Language‑Alerts",
            "Auto‑Tagging für Talent‑Pools nach Skills & Seniority",
            "Smart‑Rankings priorisieren Matches nach 360°‑Fit",
            "Interview‑Guide‑Generator (kompetenzbasiert, STAR‑Methode)",
            "Fallback‑Sourcing‑Kanäle Empfehlungen (Nischenbörsen, Community‑Foren)",
            "One‑Click‑Übergabe an Headhunter mit definierter Datentiefe",
            "Reminder‑Workflow für Hiring‑Manager‑Feedback",
            "Candidate‑Care‑Score misst Antwortzeiten & Touchpoints",
            "Ghosting‑Risiko‑Radar prognostiziert Absprungwahrscheinlichkeit",
            "Recruiter‑Leaderboard (Transparente Performance)",
            "Automated Scheduling mit Kalenderslots & Video‑Links",
            "Offer‑Benchmarking‑Wizard gegen Markt & interne Bands",
            "Snippets‑Library für schnelle Stakeholder‑Updates",
            "Mehrstufige Approval‑Matrix für Offerte & Budget",
            "ChatGPT‑Assisted Negotiation‑Tips live im Kandidatengespräch",
            "Bulk‑PDF‑Export aller Kandidatenprofile für Hiring‑Panels",
            "GDPR‑Timer für Lösch‑Fristen & Opt‑In‑Tracking",
            "Sourcing‑Funnel‑Visualization als Kanban",
            "Recruiter‑Slack‑Bot mit Daily‑Digest",
            "Social‑Listening‑Monitor (Glassdoor, Kununu)",
            "Diversity‑Heatmap gegen interne Ziele",
            "Referral‑Booster‑Widget mit Gamification",
            "Continuous‑Improvement‑Loop (Retro‑Umfragen nach jedem Hire)",
        ],
        "Unternehmen": [
            "Strategische Workforce‑Forecasts aggregiert aus allen Anfragen",
            "Data‑Lake‑Export für People‑Analytics",
            "Kosten‑Transparenz pro Cost‑Center & Hire‑Kategorie",
            "Einhaltung globaler Policies (DE&I, Remote‑First, Comp‑Ratio)",
            "Predictive‑Hiring‑Modelle für saisonale Peaks",
            "Reduziertes Time‑to‑Productivity durch passgenaue JD",
            "Weniger Fehlbesetzungen dank Skill‑Validierung vor Ausschreibung",
            "Employer‑Brand‑Stärkung durch konsistente, ansprechende Jobpages",
            "Revisionssichere Dokumentation aller Hiring‑Entscheidungen",
            "Automated SLA‑Tracking zwischen HR & Fachbereich",
            "Globale Roll‑Ups (z. B. Headcount‑Plan vs. Ist)",
            "Kapazitätsausgleich (Redeploy vs. New Hire)",
            "Reporting an Vorstand per Klick",
            "Audit‑Ready für ISO‑, SOC‑, ESG‑Prüfungen",
            "Gesicherte Datenhoheit (On‑Prem‑ oder EU‑Cloud‑Option)",
            "Legal‑Hold‑Funktion bei Streitfällen",
            "Nachfolge‑Pipeline‑Insights (Internal Mobility)",
            "Standardisierte Job‑Taxonomie für globale Vergleichbarkeit",
            "Corporate‑CI‑kompatible Vorlagen",
            "Skill‑Demand‑Trend‑Analyse für Learning & Development",
            "Bench‑Cost‑Einsparung durch bessere Forecasts",
            "M&A‑Readiness dank konsolidierter Talent‑Roadmap",
            "360°‑DE&I‑Dashboard (Gender‑Pay‑Gap, Representation, Inclusion‑Score)",
            "Automatisierte Risk‑Alerts (Überstunden, Fluktuationsrisiko)",
            "Remote‑Work‑Compliance‑Check (Zoll, Steuer, Aufenthaltsrecht)",
            "Interne Talent‑Marketplace‑Anbindung",
            "CO₂‑Footprint‑Estimator pro Recruiting‑Kampagne",
            "Governance‑Gateway für Freigaben & Richtlinien",
            "Continuous‑Quality‑Loop mit Pulse‑Surveys nach 30/90/180 Tagen",
        ],
        "Bewerber": [
            "Klare, transparente Anforderungsprofile ohne Buzzwords",
            "Realistische Gehaltsrange upfront",
            "Echtzeit‑Statusupdates via WhatsApp, E‑Mail, Portal",
            "Kalender‑Self‑Booking von Interviews",
            "Barrierefreier Bewerbungsprozess (Screen‑Reader, Tastatur‑Navi)",
            "One‑Click‑Apply dank CV‑Parsing & LinkedIn‑Import",
            "Personalisierte FAQ‑Sektion basierend auf Profil",
            "Instant‑Feedback nach Assessment (Stärken / Entwicklungsfelder)",
            "Vorhersage der Interview‑Dauer & beteiligten Personen",
            "Automatisierte Vorbereitungstipps (Tech‑Stack, Unternehmenskultur)",
            "Diversity‑Friendly Sprache minimiert Bias",
            "GDPR‑Safe Data Vault für alle Unterlagen",
            "Self‑Service‑Portal zur Dokumentenaktualisierung",
            "Option auf Anonymisierung für Blind‑Screening",
            "Geolocation‑basierte Pendel‑Zeit‑Anzeige",
            "Responsive Dark‑ & Light‑Mode",
            "Multi‑Language‑Support (mind. DE/EN/ES/FR)",
            "Talent‑Pool‑Opt‑In mit personalisierten Job‑Alerts",
            "Interview‑Reminder mit Kalendersync & Routenplaner",
            "Free‑Download persönlicher Assessment‑Reports",
            "Feedback‑Loop – Kandidat bewertet Prozess & Interviewer",
            "Referral‑Credits für erfolgreiche Empfehlungen",
            "Off‑Topic‑Opt‑Out (Keine unnötigen Daten wie Geburtsdatum)",
            "Post‑Offer‑Onboarding‑Preview (Team‑Video, Projekt‑Roadmap)",
        ],
    },
    "English": {
        "Line Manager": [
            "Immediate competency match between desired skill set and market availability",
            "Live budget calculation based on current salary benchmarks",
            "Traffic‑light alerts for unrealistic requirement combinations",
            "Automatic prioritisation of critical must‑have skills",
            "Time savings through one‑pager summary instead of long email threads",
            'Interactive scenario planning ("What does it cost if we hire senior instead of mid‑level?")',
            "Role clarity via predefined RACI responsibility matrix",
            "Custom KPI dashboards (time‑to‑hire, quality‑of‑hire, cost‑per‑hire)",
            'Version history for later audits ("who changed what and when?")',
            "Automated reminders for overdue feedback steps",
            "AI‑assisted wording suggestions for attractive benefit descriptions",
            "Skill‑gap heatmap for internal upskilling vs external hiring",
            "Instant market report (talent density, competition pressure, location attractiveness)",
            "Approval workflow with electronic signature",
            "Conflict‑free headcount tracking across multiple cost centres",
            "Self‑service reports without BI dependency",
            "Compliance check (anti‑discrimination, equal pay, EU AI Act)",
            "Benchmarks against company guidelines (salary bands, remote policy)",
            "Multilingual interface for global teams",
            "Export to PowerPoint, PDF, Excel",
            "Integration hooks (Slack, Teams, Jira) for seamless collaboration",
            '"Next best action" hints (e.g. "invite interview panel")',
            "Calendar sync to block critical deadlines",
            "Real‑time push notifications on market changes (collective agreements, etc.)",
            "Full accessibility (WCAG 2.2)",
            "Dark mode for long work sessions",
            "Mobile responsive for approvals on the go",
            "Data privacy shield separates sensitive candidate data from management info",
            "On‑demand training snippets (embedded videos, tooltips)",
            "Gamified completion tracker for motivation",
        ],
        "Recruiter": [
            "Automatic parsing of JD drafts, CVs & links",
            "Database connectivity (ATS, CRM, LinkedIn Recruiter, GitHub)",
            "Real‑time KPIs (sourcing conversion rate, pipeline velocity)",
            'AI‑generated search strings ("Boolean Builder 2.0")',
            "Candidate personas from text and profile analysis",
            "Bulk‑mail personalisation with A/B subject lines",
            "Job‑ad SEO check (Google for Jobs, Indeed)",
            "Bias detection & inclusive language alerts",
            "Auto‑tagging for talent pools by skills & seniority",
            "Smart rankings prioritise matches by 360° fit",
            "Interview guide generator (competency based, STAR method)",
            "Fallback sourcing channel recommendations (niche boards, community forums)",
            "One‑click hand‑off to head‑hunters with defined data depth",
            "Reminder workflow for hiring‑manager feedback",
            "Candidate care score measures response times & touchpoints",
            "Ghosting risk radar predicts dropout probability",
            "Recruiter leaderboard (transparent performance)",
            "Automated scheduling with calendar slots & video links",
            "Offer benchmarking wizard against market & internal bands",
            "Snippet library for quick stakeholder updates",
            "Multi‑level approval matrix for offers & budget",
            "ChatGPT‑assisted negotiation tips live during candidate calls",
            "Bulk PDF export of all candidate profiles for hiring panels",
            "GDPR timer for deletion deadlines & opt‑in tracking",
            "Sourcing funnel visualisation as Kanban",
            "Recruiter Slack bot with daily digest",
            "Social listening monitor (Glassdoor, Kununu)",
            "Diversity heatmap versus internal targets",
            "Referral booster widget with gamification",
            "Continuous improvement loop (retro surveys after each hire)",
        ],
        "Company": [
            "Strategic workforce forecasts aggregated from all requests",
            "Data‑lake export for people analytics",
            "Cost transparency per cost centre & hire category",
            "Compliance with global policies (DE&I, remote first, comp ratio)",
            "Predictive hiring models for seasonal peaks",
            "Reduced time‑to‑productivity through precise JD",
            "Fewer mis‑hires thanks to skill validation before posting",
            "Employer brand strengthening via consistent, appealing job pages",
            "Audit‑proof documentation of all hiring decisions",
            "Automated SLA tracking between HR & business",
            "Global roll‑ups (e.g. headcount plan vs actual)",
            "Capacity balancing (redeploy vs new hire)",
            "One‑click reporting to the board",
            "Audit ready for ISO, SOC, ESG examinations",
            "Secured data sovereignty (on‑prem or EU cloud option)",
            "Legal hold function in case of disputes",
            "Succession pipeline insights (internal mobility)",
            "Standardised job taxonomy for global comparability",
            "Corporate CI compliant templates",
            "Skill demand trend analysis for learning & development",
            "Bench cost savings through better forecasts",
            "M&A readiness thanks to consolidated talent roadmap",
            "360° DE&I dashboard (gender pay gap, representation, inclusion score)",
            "Automated risk alerts (overtime, turnover risk)",
            "Remote work compliance check (customs, tax, residence law)",
            "Internal talent marketplace connection",
            "CO₂ footprint estimator per recruiting campaign",
            "Governance gateway for approvals & policies",
            "HR chatbot integration for employee inquiries",
            "Continuous quality loop with pulse surveys after 30/90/180 days",
        ],
        "Candidate": [
            "Clear, transparent requirement profiles without buzzwords",
            "Realistic salary range upfront",
            "Real‑time status updates via WhatsApp, email, portal",
            "Self‑booking of interview slots via calendar",
            "Barrier‑free application process (screen reader, keyboard nav)",
            "One‑click apply via CV parsing & LinkedIn import",
            "Personalised FAQ section based on profile",
            "Instant feedback after assessment (strengths / development areas)",
            "Prediction of interview duration & participants",
            "Automated preparation tips (tech stack, company culture)",
            "Diversity‑friendly language minimises bias",
            "GDPR‑safe data vault for all documents",
            "Self‑service portal to update documents",
            "Option for anonymisation for blind screening",
            "Gamified challenges as pre‑assessment",
            "Mobile‑first flow completed in under 5 minutes",
            "Geolocation‑based commuting time display",
            "Responsive dark & light mode",
            "Multi‑language support (at least DE/EN/ES/FR)",
            "Talent pool opt‑in with personalised job alerts",
            "Interview reminders with calendar sync & route planner",
            "Live Q&A chatbot with vector search backend",
            "Status GIFs & emojis for positive candidate experience",
            "Skill match score explains fit transparently",
            "Free download of personal assessment reports",
            "Feedback loop – candidate rates process & interviewer",
            "Referral credits for successful recommendations",
            "Accessible file upload (drag & drop + cloud links)",
            "Off‑topic opt‑out (no unnecessary data like date of birth)",
            "Post‑offer onboarding preview (team video, project roadmap)",
        ],
    },
}

# ---------------------------------------------------------------------------
# Helper: Renderer
# ---------------------------------------------------------------------------


def render_benefits(title: str, items: List[str], show_top: int = 8):
    """Render top benefits plus full list in expander."""
    st.subheader(title)

    for benefit in items[:show_top]:
        st.markdown(f"• **{benefit}**")

    if len(items) > show_top:
        with st.expander(
            (
                f"Alle {len(items)} Vorteile anzeigen"
                if lang == "Deutsch"
                else f"Show all {len(items)} advantages"
            )
        ):
            for idx, benefit in enumerate(items[show_top:], start=show_top + 1):
                st.markdown(f"{idx}. {benefit}")


# ---------------------------------------------------------------------------
# Titel & Intro (sprachabhängig)
# ---------------------------------------------------------------------------

title_de = "🚀 Vorteile von **Vacalyser**"
intro_de = (
    "Wähle deine Perspektive und entdecke die spezifischen Mehrwerte. "
    "Nutze die *Alle Vorteile anzeigen*-Schaltfläche, um die komplette Liste zu sehen."
)

title_en = "🚀 Advantages of **Vacalyser**"
intro_en = (
    "Choose your perspective and discover the specific benefits. "
    "Use the *Show all advantages*-button to reveal the full list."
)

st.title(title_de if lang == "Deutsch" else title_en)

st.markdown(intro_de if lang == "Deutsch" else intro_en)

# ---------------------------------------------------------------------------
# Layout: Tabs
# ---------------------------------------------------------------------------

tab_labels_de = ["👩‍💼 Line Manager", "🧑‍💻 Recruiter", "🏢 Unternehmen", "🙋 Bewerber"]

tab_labels_en = ["👩‍💼 Line Manager", "🧑‍💻 Recruiter", "🏢 Company", "🙋 Candidate"]

labels = tab_labels_de if lang == "Deutsch" else tab_labels_en

tabs = st.tabs(labels)

with tabs[0]:
    render_benefits(labels[0], benefits[lang]["Line Manager"])

with tabs[1]:
    render_benefits(labels[1], benefits[lang]["Recruiter"])

with tabs[2]:
    label_key = "Unternehmen" if lang == "Deutsch" else "Company"
    render_benefits(labels[2], benefits[lang][label_key])

with tabs[3]:
    label_key = "Bewerber" if lang == "Deutsch" else "Candidate"
    render_benefits(labels[3], benefits[lang][label_key])

# ---------------------------------------------------------------------------
# Footer Hinweis
# ---------------------------------------------------------------------------
footer_de = "Hinweis: Passe die Listen in *pages/advantages.py* frei an, um sie zu kürzen oder zu erweitern."
footer_en = (
    "Note: Adjust the lists in *pages/advantages.py* freely to shorten or extend them."
)

st.caption(footer_de if lang == "Deutsch" else footer_en)
