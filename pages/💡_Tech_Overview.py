# pages/tech_overview.py
"""Streamlit-Seite: Technology Deep Dive & Wizard Flow

Für IT‑Spezialisten und Entscheider bietet diese Seite einen kompakten, aber
technisch fundierten Überblick über den *Vacalyser*-Stack sowie eine visuelle
Darstellung des mehrstufigen Wizard‑Flows (Discovery‑Process).
Ein Sprach‑ und Zielgruppenumschalter sorgt dafür, dass Texte sowohl für ein
Fach‑Publikum (Tech‑interessiert/Tech‑savvy) als auch für nicht‑technische
Stakeholder (Allgemein verständlich/General public) optimal angepasst werden.
"""

import streamlit as st

# ---------------------------------------------------------------------------
# Language & audience toggle
# ---------------------------------------------------------------------------
lang_label = st.radio(
    "🌐 Sprache / Language",
    ("Deutsch", "English"),
    horizontal=True,
    key="tech_lang",
)
lang = "de" if lang_label == "Deutsch" else "en"
audience = st.radio(
    "🎯 Zielgruppe / Audience",
    (
        ("Tech-interessiert", "Allgemein verständlich")
        if lang == "de"
        else ("Tech-savvy", "General public")
    ),
    horizontal=True,
    key="audience",
)

TECH_AUDIENCE = "Tech-interessiert" if lang == "de" else "Tech-savvy"

# ---------------------------------------------------------------------------
# Technology catalogue
# ---------------------------------------------------------------------------
tech_info = {
    "Deutsch": {
        "Tech-interessiert": [
            (
                "Retrieval-Augmented Generation (RAG)",
                "FAISS bzw. künftig ChromaDB/Weaviate liefern Vektor‑Suche über mehr als 400 000 ESCO‑Skills und Domain‑Korpora; LangChain orchestriert die RAG‑Pipeline.",
            ),
            (
                "LangChain Agents & OpenAI Function Calling",
                "Deterministische Tool‑Aufrufe (PDF‑Parser, ESCO‑Lookup, Markdown‑Renderer) mittels JSON‑Schemas für robustes Error‑Handling.",
            ),
            (
                "Embedding‑Model",
                "OpenAI *text-embedding-3-small* (8 k Dim); selbstgehostete Alternative *e5-large-v2* ist vorbereitet.",
            ),
            (
                "Streaming Responses",
                "Tokenweises UI‑Streaming (< 300 ms TTFB) für flüssige Nutzer‑Erfahrung.",
            ),
            (
                "CI/CD Pipeline",
                "GitHub Actions → Docker → Terraform; Canary‑Deployments auf Kubernetes mit automatischem Rollback.",
            ),
            (
                "Observability & Kosten‑Tracking",
                "OpenTelemetry Tracing + Prometheus/Grafana; Token‑Kosten pro Request im UI sichtbar.",
            ),
            (
                "Security Layer",
                "OIDC‑basiertes Secrets‑Management und zweistufige Rollenlogik (Recruiter vs. Admin).",
            ),
            (
                "Event‑Driven Wizard Flow",
                "Finite‑State‑Machine triggert dynamische Fragen und speichert Zwischenergebnisse als JSON‑Graph.",
            ),
            (
                "Infrastructure as Code",
                "Vollständige Cloud‑Provisionierung in Terraform 1.7 mit Drift‑Detection.",
            ),
        ],
        "Allgemein verständlich": [
            (
                "Künstliche Intelligenz",
                "Vacalyser nutzt modernste KI, um Stellenanforderungen präzise zu verstehen und passende Kompetenzen vorzuschlagen.",
            ),
            (
                "Schlaue Suche",
                "Eine Spezial‑Suche findet blitzschnell relevante Fähigkeiten und Aufgaben.",
            ),
            (
                "Fließende Antworten",
                "Antworten erscheinen Stück für Stück – Wartezeiten verkürzen sich.",
            ),
            (
                "Automatische Updates",
                "Neue Versionen werden im Hintergrund eingespielt, ohne Ausfallzeiten.",
            ),
            (
                "Sicherheit & Datenschutz",
                "Aktuelle Standards schützen vertrauliche Daten konsequent.",
            ),
        ],
    },
    "English": {
        "Tech-savvy": [
            (
                "Retrieval-Augmented Generation (RAG)",
                "FAISS – future upgrade to ChromaDB/Weaviate – provides vector search across 400 k+ ESCO skills & domain corpora, orchestrated via LangChain.",
            ),
            (
                "LangChain Agents & OpenAI Function Calling",
                "Deterministic tool invocation (PDF parser, ESCO lookup, Markdown renderer) using strict JSON schemas for resilient error handling.",
            ),
            (
                "Embedding Model",
                "OpenAI *text-embedding-3-small* (8 k dim); self‑hosted fallback *e5-large-v2* prepared.",
            ),
            (
                "Streaming Responses",
                "Sub‑300 ms TTFB with token‑level UI streaming for a snappy UX.",
            ),
            (
                "CI/CD Pipeline",
                "GitHub Actions → Docker → Terraform; canary deployments on Kubernetes with auto‑rollback.",
            ),
            (
                "Observability & Cost Governance",
                "OpenTelemetry tracing + Prometheus/Grafana; token cost per request surfaced in the UI.",
            ),
            (
                "Security Layer",
                "OIDC‑backed secret management and dual role model (Recruiter vs. Admin).",
            ),
            (
                "Event‑Driven Wizard Flow",
                "Finite state machine triggers dynamic questions and stores interim results as a JSON graph.",
            ),
            (
                "Infrastructure as Code",
                "Full cloud provisioning in Terraform 1.7 with automatic drift detection.",
            ),
        ],
        "General public": [
            (
                "Artificial Intelligence",
                "Vacalyser uses cutting‑edge AI to understand job requirements and suggest matching skills.",
            ),
            (
                "Smart Search",
                "A specialised search engine instantly finds relevant skills and tasks.",
            ),
            ("Live Answers", "Replies appear gradually, so you don’t have to wait."),
            (
                "Automatic Updates",
                "New versions are rolled out silently with no downtime.",
            ),
            (
                "Security & Privacy",
                "Modern standards keep your data safe at every step.",
            ),
        ],
    },
}

# ---------------------------------------------------------------------------
# Wizard flow definition
# ---------------------------------------------------------------------------
wizard_steps = [
    ("Intake", "Job‑Titel & Dokumente" if lang == "de" else "Job title & docs"),
    ("Parse", "AI‑Parsing"),
    ("Enrich", "ESCO‑Mapping"),
    ("QA", "Dynamic Q&A"),
    ("Draft", "Profil‑Entwurf" if lang == "de" else "Draft profile"),
    ("Review", "Freigabe" if lang == "de" else "Review"),
    ("Export", "Export (PDF/MD)"),
]


def render_wizard_graph() -> None:
    dot = (
        "digraph wizard {\n"
        "  rankdir=LR;\n"
        '  node [shape=box style="rounded,filled" fontname=Helvetica color=#5b8def fillcolor=#eef4ff];\n'
    )
    for step, label in wizard_steps:
        dot += f'  {step} [label="{label}"];\n'
    for idx in range(len(wizard_steps) - 1):
        dot += f"  {wizard_steps[idx][0]} -> {wizard_steps[idx + 1][0]};\n"
    dot += "}"
    st.graphviz_chart(dot)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
if audience == TECH_AUDIENCE and lang == "de":
    title = "🛠️ Technischer Deep Dive"
elif audience == TECH_AUDIENCE:
    title = "🛠️ Technology Deep Dive"
elif lang == "de":
    title = "🛠️ Technologischer Überblick"
else:
    title = "🛠️ Technology Overview"

st.title(title)

intro = (
    "Nachfolgend findest du die Schlüsseltechnologien, die Vacalyser antreiben, "
    "sowie eine Grafik, die den Discovery‑Prozess Schritt für Schritt veranschaulicht."
    if lang == "de"
    else "Below you can explore the core technologies powering Vacalyser together with a graph "
    "illustrating each step of the discovery process."
)

st.markdown(intro)

# ─── Technology cards ───
for tech, desc in tech_info[lang_label][audience]:
    st.markdown(f"### 🔹 {tech}\n{desc}")

# ─── Wizard flow graph for tech audience ───
if audience == TECH_AUDIENCE:
    st.divider()
    st.markdown(
        "#### 🔄 Wizard‑Flow & State Machine"
        if lang == "de"
        else "#### 🔄 Wizard Flow & State Machine"
    )
    render_wizard_graph()

st.divider()

st.info(
    "Die gezeigte Architektur ist modular erweiterbar und bildet eine zukunftssichere Basis für hochskalierbare Recruiting‑Workflows."
    if lang == "de"
    else "The presented stack is modular and future‑proof, enabling highly scalable recruiting workflows with minimal operational overhead."
)
