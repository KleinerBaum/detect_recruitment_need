import asyncio
import importlib.util
import os
import sys
from pathlib import Path
import types


def load_tool_module():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    spec = importlib.util.spec_from_file_location("tool", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    os.environ.setdefault("OPENAI_API_KEY", "test")
    spec.loader.exec_module(module)
    return module


def test_parse_skill_list_variants():
    tool = load_tool_module()
    assert tool.parse_skill_list("Python, SQL\nExcel") == ["Python", "SQL", "Excel"]
    assert tool.parse_skill_list(["Docker", "Kubernetes"]) == ["Docker", "Kubernetes"]


async def dummy_create(*args, **kwargs):
    content = '{"skills": ["SkillA", "SkillB"]}'
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_suggest_hard_skills(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    data = {"job_title": "Engineer"}
    out = asyncio.run(tool.suggest_hard_skills(data))
    assert out == ["SkillA", "SkillB"]
