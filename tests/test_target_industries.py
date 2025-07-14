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
    prompts.append(kwargs["messages"][1]["content"])
    if "benefit" in prompts[-1]:
        content = '{"benefits": ["B1"]}'
    else:
        content = '{"items": ["D1"]}'
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])


def test_prompts_include_target_industries(monkeypatch):
    tool = load_tool_module()
    global prompts
    prompts = []
    monkeypatch.setattr(tool.client.chat.completions, "create", dummy_create)
    data = {
        "job_title": "Engineer",
        "industry": "IT",
        "target_industries": ["Finance", "Healthcare"],
    }
    asyncio.run(tool.suggest_benefits_competitors(data, 5))
    asyncio.run(tool.suggest_client_difficulties(data, 5))
    assert "Finance" in prompts[0]
    assert "Finance" in prompts[1]


def test_ideal_candidate_profile_mapping():
    tool = load_tool_module()
    assert tool.KEY_TO_STEP["ideal_candidate_profile"] == "SUMMARY"
