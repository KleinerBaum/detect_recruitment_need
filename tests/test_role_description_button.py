import importlib.util
import os
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest


def load_tool_module():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    spec = importlib.util.spec_from_file_location(
        "Recruitment_Need_Analysis_Tool", path
    )
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_generate_role_description_button(monkeypatch, tmp_path):
    tool = load_tool_module()

    script = tmp_path / "small_app.py"
    script.write_text(
        """
import streamlit as st
import asyncio
from streamlit import session_state as ss
from Recruitment_Need_Analysis_Tool import suggest_role_description
ss.setdefault('data', {'job_title': 'Engineer'})
if st.button('Generate Role Description', key='gen_role_desc'):
    with st.spinner('Generating …'):
        ss['data']['role_description'] = asyncio.run(suggest_role_description(ss['data']))
if ss['data'].get('role_description'):
    st.text_area('Role Description', ss['data']['role_description'], key='ROLE_role_description', disabled=True)
"""
    )

    async def dummy(data):
        return "desc"

    monkeypatch.setattr(tool, "suggest_role_description", dummy)
    sys.modules["Recruitment_Need_Analysis_Tool"] = tool

    at = AppTest.from_file(str(script), default_timeout=10)
    at.run()
    at.button(key="gen_role_desc").click()
    at.run()

    assert at.session_state["data"]["role_description"] == "desc"
    assert at.text_area(key="ROLE_role_description").value == "desc"
