"""Input validation helpers."""

from __future__ import annotations

import re
from typing import Callable, Dict

from dateutil import parser as dateparser
from ui_forms import EMAIL_RE


def parse_date_value(value: str) -> str | None:
    """Parse a user provided date string to ISO format."""
    try:
        parsed = dateparser.parse(value)
    except Exception:
        return None
    if not parsed:
        return None
    return parsed.date().isoformat()


def validate_email(value: str) -> str | None:
    """Return trimmed email if it matches EMAIL_RE."""
    val = value.strip()
    return val if re.match(EMAIL_RE, val) else None


def validate_phone(value: str) -> str | None:
    """Validate phone numbers with optional '+' and digits."""
    digits = value.strip().replace(" ", "").replace("-", "")
    return value.strip() if re.match(r"^\+?\d+$", digits) else None


# Mapping of field keys to validator functions
VALIDATORS: Dict[str, Callable[[str], str | None]] = {
    "date_of_employment_start": parse_date_value,
    "contract_end_date": parse_date_value,
    "contact_email": validate_email,
    "recruitment_contact_email": validate_email,
    "line_manager_email": validate_email,
    "hr_poc_email": validate_email,
    "recruitment_contact_phone": validate_phone,
}
