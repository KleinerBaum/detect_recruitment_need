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


async def dummy_create(*args, **kwargs):
    msg = types.SimpleNamespace(content='{"tasks": ["T1", "T2"]}')
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_task_functions(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    ai = asyncio.run(tool.suggest_tasks({"job_title": "Dev"}))
    assert ai == ["T1", "T2"]

    monkeypatch.setattr(tool, "search_occupations", lambda q, limit=1: [{"uri": "u"}])
    monkeypatch.setattr(
        tool,
        "get_skills_for_occupation",
        lambda uri, limit=10: [{"title": "A"}, {"label": "B"}],
    )
    esco = tool.get_esco_tasks("dev")
    assert esco == ["A", "B"]
