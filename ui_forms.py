from __future__ import annotations

import re
import streamlit as st

EMAIL_RE = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"


def email_input(label: str, key: str) -> str:
    """Render a validated email input field.

    Parameters
    ----------
    label : str
        Visible label for the input field.
    key : str
        Session state key for Streamlit.

    Returns
    -------
    str
        The current value of the text input.
    """

    val = st.text_input(label, key=key)
    if val and not re.match(EMAIL_RE, val):
        st.error("Ungültige E-Mail Adresse")
    return val
