import asyncio
import importlib.util
import os
from pathlib import Path
import types


def load_tool_module():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    spec = importlib.util.spec_from_file_location("tool", path)
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    spec.loader.exec_module(module)
    return module


async def dummy_create(*args, **kwargs):
    msg = types.SimpleNamespace(content="profile")
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_generate_ideal_candidate_profile(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    res = asyncio.run(
        tool.generate_ideal_candidate_profile({}, [("task", 2)], [("skill", 3)])
    )
    assert res == "profile"
