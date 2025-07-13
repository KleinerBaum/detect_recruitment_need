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
    content = '{"benefits": ["B1", "B2"]}'
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_suggest_benefits_functions(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    data = {"job_title": "Engineer", "city": "Metropolis"}
    out1 = asyncio.run(tool.suggest_benefits_by_title(data, 5))
    out2 = asyncio.run(tool.suggest_benefits_by_location(data, 5))
    out3 = asyncio.run(tool.suggest_benefits_competitors(data, 5))
    assert out1 == ["B1", "B2"]
    assert out2 == ["B1", "B2"]
    assert out3 == ["B1", "B2"]
