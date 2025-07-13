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


async def dummy_create_text(*args, **kwargs):
    msg = types.SimpleNamespace(content="desc")
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


async def dummy_create_json(*args, **kwargs):
    msg = types.SimpleNamespace(content='{"items": ["A", "B"]}')
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_suggest_role_description(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create_text)

    async def dummy_search(*args, **kwargs):
        return ["ctx"]

    monkeypatch.setattr(tool.vector_store, "search", dummy_search)
    res = asyncio.run(tool.suggest_role_description({"job_title": "Engineer"}))
    assert res == "desc"


def test_recruitment_and_questions(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create_json)
    steps = asyncio.run(tool.suggest_recruitment_steps({"job_title": "Engineer"}, 3))
    questions = asyncio.run(
        tool.suggest_interview_questions({"job_title": "Engineer"}, 2)
    )
    assert steps == ["A", "B"]
    assert questions == ["A", "B"]
