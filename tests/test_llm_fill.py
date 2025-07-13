import asyncio
import types
import importlib.util
import os
from pathlib import Path


def load_tool_module():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    spec = importlib.util.spec_from_file_location("tool", path)
    module = importlib.util.module_from_spec(spec)
    os.environ.setdefault("OPENAI_API_KEY", "test")
    spec.loader.exec_module(module)
    return module


async def dummy_create(*args, **kwargs):
    return types.SimpleNamespace(
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="{}"))]
    )


async def dummy_search(query, top_k=3):
    dummy_search.called = query
    return ["ctx"]


def test_llm_fill_includes_context(monkeypatch):
    tool = load_tool_module()
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    monkeypatch.setattr(tool.vector_store, "search", dummy_search)
    asyncio.run(tool.llm_fill(["job_title"], "some text"))
    assert dummy_search.called == "some text"[:1000]


async def dummy_create_retry(*args, **kwargs):
    dummy_create_retry.calls += 1
    if dummy_create_retry.calls == 1:
        content = "{}"
    else:
        content = '{"job_title": {"value": "Engineer", "confidence": 0.9}}'
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_llm_fill_retries_on_missing(monkeypatch):
    tool = load_tool_module()
    dummy_create_retry.calls = 0
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create_retry)
    monkeypatch.setattr(tool.vector_store, "search", dummy_search)
    res = asyncio.run(tool.llm_fill(["job_title"], "text"))
    assert dummy_create_retry.calls == 2
    assert res["job_title"].value == "Engineer"
