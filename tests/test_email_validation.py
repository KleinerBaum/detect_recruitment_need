import importlib.util
import re
from pathlib import Path

import pytest


def load_forms_module():
    path = Path(__file__).resolve().parents[1] / "ui_forms.py"
    spec = importlib.util.spec_from_file_location("forms", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "email",
    [
        "user@example.com",
        "first.last@domain.co",
        "name+alias@sub.domain.org",
    ],
)
def test_email_regex_valid(email):
    forms = load_forms_module()
    assert re.match(forms.EMAIL_RE, email)


@pytest.mark.parametrize(
    "email",
    [
        "userexample.com",
        "bad@domain",
        "no space@domain.com",
    ],
)
def test_email_regex_invalid(email):
    forms = load_forms_module()
    assert not re.match(forms.EMAIL_RE, email)
