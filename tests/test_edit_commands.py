import importlib.util
import os
import sys
from pathlib import Path


def load_app_module():
    path = Path(__file__).resolve().parents[1] / "app.py"
    sys.path.insert(0, str(path.parent))
    os.environ.setdefault("OPENAI_API_KEY", "test")
    import streamlit as st

    st.secrets = {"OPENAI_API_KEY": "test"}
    spec = importlib.util.spec_from_file_location("app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_parse_edit_command_colon():
    app = load_app_module()
    assert app.parse_edit_command("job_title: Senior Engineer") == (
        "job_title",
        "Senior Engineer",
    )


def test_parse_edit_command_change():
    app = load_app_module()
    text = "Please change the company name to Tech Corp"
    assert app.parse_edit_command(text) == ("company_name", "Tech Corp")
