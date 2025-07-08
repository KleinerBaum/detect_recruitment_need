import asyncio
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

    monkeypatch.setattr(tool, "llm_fill", dummy_fill)
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
