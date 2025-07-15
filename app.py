"""Simple chat interface for Vacalyser.

This module provides a minimal chat UI in Streamlit. It initializes a
conversation with a welcome message and defines structures for a multi-step
dialog. The assistant guides the user through a sequence of questions to build
a recruitment profile.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Coroutine, Dict, List, Optional, Tuple

import asyncio
import re

import streamlit as st
from validation_utils import (
    VALIDATORS,
    parse_date_value,
    validate_email,
    validate_phone,
)

from file_tools import extract_text_from_file
from services.langchain_chain import fetch_url_text
from Recruitment_Need_Analysis_Tool import extract

WELCOME_MESSAGE = (
    "Hello! I'm here to help gather all the details for your recruitment needs. "
    "I'll guide you through some questions to create a comprehensive job profile."
)

# ---------------------------------------------------------------------------
# Dialog Schema
# ---------------------------------------------------------------------------

SECTION_FIELDS: OrderedDict[str, List[Tuple[str, str]]] = OrderedDict(
    [
        (
            "Basic Info",
            [
                ("job_title", "What is the job title for the position?"),
                ("employment_type", "Is it a full-time or part-time role?"),
            ],
        ),
        (
            "Company Info",
            [
                ("company_name", "What is the company name?"),
                ("company_location", "Where is the position located?"),
            ],
        ),
        (
            "Role",
            [
                ("summary", "Give a short description of the role."),
            ],
        ),
        (
            "Skills",
            [
                ("skills", "List key skills required."),
            ],
        ),
        (
            "Contacts",
            [
                ("contact_email", "What email should candidates use to apply?"),
            ],
        ),
    ]
)

# Short explanations to show with each question
FIELD_TIPS: Dict[str, str] = {
    "job_title": "This helps us clearly define the position.",
    "employment_type": "Clarifies the expected working hours.",
    "company_name": "Lets candidates know who they will work for.",
    "company_location": "Important for commuting or relocation decisions.",
    "summary": "Gives candidates an overview of the role.",
    "skills": "Highlights the key competencies you need.",
    "contact_email": "So applicants can reach out with questions.",
}

FIELD_FLOW: List[Tuple[str, str, str]] = [
    (section, key, prompt)
    for section, fields in SECTION_FIELDS.items()
    for key, prompt in fields
]

FIELD_ALIASES: Dict[str, List[str]] = {
    key: [key.replace("_", " ")] for _, key, _ in FIELD_FLOW
}


def parse_edit_command(text: str) -> Optional[Tuple[str, str]]:
    """Return ``(field, value)`` if the user wants to edit a field."""

    lower = text.lower().strip()

    if ":" in text:
        field_part, value = text.split(":", 1)
        field_part = field_part.strip().lower()
        for key, aliases in FIELD_ALIASES.items():
            if field_part == key or field_part in aliases:
                return key, value.strip()

    if any(word in lower for word in ["change", "edit", "update"]):
        for key, aliases in FIELD_ALIASES.items():
            for alias in [key] + aliases:
                if alias in lower:
                    match = re.search(r"(?:to|:)\s*(.+)$", text, re.IGNORECASE)
                    if match:
                        return key, match.group(1).strip()

    return None


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run async coroutine from sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return loop.run_until_complete(coro)


def update_step_from_data() -> None:
    """Set step to the first unanswered field."""
    for i, (_, key, _) in enumerate(FIELD_FLOW):
        if not st.session_state.data.get(key):
            st.session_state.step = i
            return
    st.session_state.step = len(FIELD_FLOW)


def _format_summary(data: Dict[str, Any]) -> str:
    """Return collected data as a markdown bullet list."""

    lines: List[str] = []
    for _, key, _ in FIELD_FLOW:
        val = data.get(key)
        if val:
            label = key.replace("_", " ").title()
            if isinstance(val, list):
                val = ", ".join(str(v) for v in val)
            lines.append(f"* **{label}:** {val}")
    return "\n".join(lines)


def display_final_summary() -> None:
    """Append a final summary message to the chat history."""

    if st.session_state.summary_shown:
        return
    summary = _format_summary(st.session_state.data)
    msg = (
        "Thank you! I've gathered all the information. "
        "Here's a summary of the job profile you provided:\n\n"
        + summary
        + "\n\nDoes everything look correct? "
        "If so, you can confirm, or let me know if any detail needs changing. ✅"
    )
    st.session_state.messages.append({"role": "assistant", "content": msg})
    st.session_state.summary_shown = True


def autofill_from_source(file: Any | None, url: str) -> None:
    """Extract data from a document or URL and prefill fields."""
    text = ""
    if file is not None:
        text = extract_text_from_file(file.getvalue(), file.type)
    elif url:
        text = run_async(fetch_url_text(url))

    if not text:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Sorry, I couldn't read that source. Let's continue manually.",
            }
        )
        return

    extracted = run_async(extract(text))
    summary: list[str] = []
    for key, res in extracted.items():
        if res.value:
            st.session_state.data[key] = res.value
            summary.append(f"**{key}**: {res.value}")

    if summary:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "I've extracted the following details:\n"
                + "\n".join(summary),
            }
        )
    update_step_from_data()
    ask_current_question()


def init_session() -> None:
    """Initialize session state for the conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.setdefault("data", {})
    st.session_state.setdefault("step", 0)
    st.session_state.setdefault("summary_shown", False)


def ask_current_question() -> None:
    """Append the next question to the chat history."""
    index = st.session_state.step
    if index >= len(FIELD_FLOW):
        display_final_summary()
        return

    section, key, prompt = FIELD_FLOW[index]
    if index == 0 or FIELD_FLOW[index - 1][0] != section:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Now let's discuss {section} details.",
            }
        )

    try:
        prompt = prompt.format(**st.session_state.data)
    except Exception:
        pass
    tip = FIELD_TIPS.get(key)
    if tip:
        prompt = f"{prompt} ({tip})"
    st.session_state.messages.append({"role": "assistant", "content": prompt})


def main() -> None:
    """Render the chat UI with persistent assistant welcome message."""
    st.title("Vacalyser Chat")
    init_session()

    if st.session_state.step == 0 and not st.session_state.get("autofill_done"):
        with st.chat_message("assistant"):
            st.write(
                "If you have a job description file or a relevant webpage, you can provide it now for autofill:"
            )
            file = st.file_uploader(
                "Upload PDF or DOCX", type=["pdf", "docx"], key="job_file"
            )
            url = st.text_input("Or enter a URL", key="job_url")
            if st.button("Autofill", key="autofill_btn"):
                with st.spinner("Extracting…"):
                    autofill_from_source(file, url)
                st.session_state.autofill_done = True
                st.rerun()

    if st.session_state.step == 0 and len(st.session_state.messages) == 1:
        ask_current_question()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Your response…")
    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        edit = parse_edit_command(user_input)
        if edit:
            field, value = edit
            st.session_state.data[field] = value
            st.session_state.summary_shown = False
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": f"Updated {field.replace('_', ' ').title()} to {value}.",
                }
            )
            index = st.session_state.step
            if index < len(FIELD_FLOW) and FIELD_FLOW[index][1] == field:
                st.session_state.step += 1
            ask_current_question()
            st.stop()

        index = st.session_state.step
        if index < len(FIELD_FLOW):
            _, key, _ = FIELD_FLOW[index]
            validator = VALIDATORS.get(key)
            value = user_input
            if validator:
                cleaned = validator(user_input)
                if cleaned is None:
                    error = "Please provide a valid value."
                    if validator is parse_date_value:
                        error = (
                            "The date format wasn't recognized. "
                            "Could you provide it as YYYY-MM-DD?"
                        )
                    elif validator is validate_email:
                        error = (
                            "That email doesn't look valid. "
                            "Please enter a correct address (e.g., name@domain.com)."
                        )
                    elif validator is validate_phone:
                        error = (
                            "Please enter a valid phone number "
                            "(digits and optional '+' only)."
                        )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error}
                    )
                    ask_current_question()
                    st.stop()
                value = cleaned
            st.session_state.data[key] = value
            st.session_state.step += 1
        else:
            confirm = user_input.strip().lower()
            if confirm in {"done", "yes", "confirm", "ok", "looks good"}:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Great! I've recorded all the information.",
                    }
                )
                st.stop()
            else:
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "Please specify changes or type 'done' to finish.",
                    }
                )

        ask_current_question()


if __name__ == "__main__":
    main()
