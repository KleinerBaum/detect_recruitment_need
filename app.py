"""Simple chat interface for Vacalyser.

This module provides a minimal chat UI in Streamlit. It initializes a
conversation with a welcome message and defines structures for a multi-step
dialog. User input handling will be added in later steps.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import List, Tuple

import streamlit as st

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

FIELD_FLOW: List[Tuple[str, str, str]] = [
    (section, key, prompt)
    for section, fields in SECTION_FIELDS.items()
    for key, prompt in fields
]


def init_session() -> None:
    """Initialize session state for the conversation."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]
    st.session_state.setdefault("data", {})
    st.session_state.setdefault("step", 0)


def ask_current_question() -> None:
    """Append the next question to the chat history."""
    index = st.session_state.step
    if index >= len(FIELD_FLOW):
        return
    section, key, prompt = FIELD_FLOW[index]
    if index == 0 or FIELD_FLOW[index - 1][0] != section:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": f"Now let's discuss {section} details.",
            }
        )
    st.session_state.messages.append({"role": "assistant", "content": prompt})


def main() -> None:
    """Render the chat UI with persistent assistant welcome message."""
    st.title("Vacalyser Chat")
    init_session()

    if st.session_state.step == 0 and len(st.session_state.messages) == 1:
        ask_current_question()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Your response…")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)


if __name__ == "__main__":
    main()
