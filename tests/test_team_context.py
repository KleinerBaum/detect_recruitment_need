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
    content = '{"items": ["A", "B"]}'
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_team_context_suggestions(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    data = {"industry": "IT"}
    out1 = asyncio.run(tool.suggest_team_challenges(data))
    out2 = asyncio.run(tool.suggest_client_difficulties(data))
    out3 = asyncio.run(tool.suggest_recent_team_changes(data))
    out4 = asyncio.run(tool.suggest_tech_stack(data))
    assert out1 == ["A", "B"]
    assert out2 == ["A", "B"]
    assert out3 == ["A", "B"]
    assert out4 == ["A", "B"]
