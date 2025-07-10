import asyncio
import types
import importlib.util
import os
import sys
from pathlib import Path


def load_tool_module():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    spec = importlib.util.spec_from_file_location("tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    os.environ.setdefault("OPENAI_API_KEY", "test")
    spec.loader.exec_module(module)
    return module


def test_extract_fallback_patterns(monkeypatch):
    tool = load_tool_module()

    async def dummy_fill(missing, text):
        return {}

    async def dummy_validate(data):
        return {}

    monkeypatch.setattr(tool, "llm_fill", dummy_fill)
    monkeypatch.setattr(tool, "llm_validate", dummy_validate)
    text = (
        "Wir suchen bei Example GmbH einen Senior Data Scientist (Vollzeit, unbefristet) "
        "in Berlin. Das Gehalt beträgt 60000 - 70000 EUR."
    )
    result = asyncio.run(tool.extract(text))
    assert result["company_name"].value == "Example GmbH"
    assert result["employment_type"].value.lower() == "vollzeit"
    assert result["contract_type"].value.lower() == "unbefristet"
    assert result["seniority_level"].value.lower().startswith("senior")
    assert result["salary_range"].value == "60000 - 70000"


def test_extract_label_company(monkeypatch):
    tool = load_tool_module()

    async def dummy_fill(missing, text):
        return {}

    async def dummy_validate(data):
        return {}

    monkeypatch.setattr(tool, "llm_fill", dummy_fill)
    monkeypatch.setattr(tool, "llm_validate", dummy_validate)
    text = "Company Name: Example GmbH\nOrt: Hamburg"
    result = asyncio.run(tool.extract(text))
    assert result["company_name"].value == "Example GmbH"


def test_extract_bullet_prefix(monkeypatch):
    tool = load_tool_module()

    async def dummy_fill(missing, text):
        return {}

    async def dummy_validate(data):
        return {}

    monkeypatch.setattr(tool, "llm_fill", dummy_fill)
    monkeypatch.setattr(tool, "llm_validate", dummy_validate)

    text = "- Company Name: Example GmbH\n- City: Hamburg\n* Job Title: Engineer"
    result = asyncio.run(tool.extract(text))
    assert result["company_name"].value == "Example GmbH"
    assert result["city"].value == "Hamburg"
    assert result["job_title"].value == "Engineer"


def test_extract_synonym_labels(monkeypatch):
    tool = load_tool_module()

    async def dummy_fill(missing, text):
        return {}

    async def dummy_validate(data):
        return {}

    monkeypatch.setattr(tool, "llm_fill", dummy_fill)
    monkeypatch.setattr(tool, "llm_validate", dummy_validate)

    text = "Jobtitel: Data Engineer\n" "Beschäftigungsverhältnis: Teilzeit"
    result = asyncio.run(tool.extract(text))
    assert result["job_title"].value == "Data Engineer"
    assert result["employment_type"].value.lower() == "teilzeit"


def test_llm_validate(monkeypatch):
    tool = load_tool_module()

    async def dummy_create(*args, **kwargs):
        content = '{"company_name": {"value": "ACME GmbH", "confidence": 0.9}}'
        msg = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    data = {"company_name": tool.ExtractResult("ACME GmbH", 0.6)}
    res = asyncio.run(tool.llm_validate(data))
    assert res["company_name"].value == "ACME GmbH"
    assert res["company_name"].confidence == 0.9
