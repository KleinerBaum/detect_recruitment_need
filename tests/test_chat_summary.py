import importlib.util
import os
import sys
import types
from pathlib import Path


def load_app_module():
    path = Path(__file__).resolve().parents[1] / "app.py"
    sys.path.insert(0, str(path.parent))
    os.environ.setdefault("OPENAI_API_KEY", "test")
    import streamlit as st

    st.secrets = {"OPENAI_API_KEY": "test"}
    spec = importlib.util.spec_from_file_location("app", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_format_summary():
    app = load_app_module()
    data = {"job_title": "Dev", "employment_type": "Full-time"}
    text = app._format_summary(data)
    assert "**Job Title:**" in text
    assert "Dev" in text


def test_display_final_summary(monkeypatch):
    app = load_app_module()
    state = types.SimpleNamespace(
        data={"job_title": "Dev"}, messages=[], summary_shown=False
    )
    monkeypatch.setattr(app, "st", types.SimpleNamespace(session_state=state))
    app.display_final_summary()
    assert state.summary_shown is True
    assert state.messages
