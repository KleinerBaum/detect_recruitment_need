"""Simple chat interface for Vacalyser."""

from __future__ import annotations

import streamlit as st

WELCOME_MESSAGE = (
    "Hello! I'm here to help gather all the details for your recruitment needs. "
    "I'll guide you through some questions to create a comprehensive job profile."
)


def init_session() -> None:
    """Initialize chat history with a single assistant welcome message."""
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": WELCOME_MESSAGE}]


def main() -> None:
    """Render the chat UI with persistent assistant welcome message."""
    st.title("Vacalyser Chat")
    init_session()

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
