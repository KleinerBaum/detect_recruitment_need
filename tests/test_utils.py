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


def test_safe_json_load_valid_and_malformed():
    tool = load_tool_module()
    assert tool.safe_json_load('{"a": 1}') == {"a": 1}
    assert tool.safe_json_load('{"a": 1,}') == {"a": 1}
    assert tool.safe_json_load("{'a': 1}") == {"a": 1}
    assert tool.safe_json_load('{"a": 1') == {"a": 1}


def test_safe_json_load_migrates_primary():
    tool = load_tool_module()
    res = tool.safe_json_load('{"primary_responsibilities": "foo"}')
    assert "primary_responsibilities" not in res
    assert res["key_responsibilities"] == "foo"


def test_pattern_search_confidence_and_value():
    tool = load_tool_module()
    res = tool.pattern_search("Name: Alice", "name", r"(?P<name>.+)")
    assert res is not None
    assert res.value == "Alice"
    assert res.confidence == 0.9


def test_http_text_handles_http_error(monkeypatch):
    tool = load_tool_module()

    def raise_error(*args, **kwargs):
        raise tool.httpx.HTTPError("fail")

    monkeypatch.setattr(tool.httpx, "get", raise_error)
    assert tool.http_text("http://example.com") == ""


def test_html_text_removes_scripts():
    tool = load_tool_module()
    html = "<p>Hello</p><script>alert(1)</script>"
    assert tool.html_text(html) == "Hello"


def test_sanitize_value():
    tool = load_tool_module()
    assert tool.sanitize_value("  Foo \n") == "Foo"
    assert tool.sanitize_value(12.0) == "12"
    assert tool.sanitize_value(None) is None


def test_sync_remote_policy_hybrid():
    tool = load_tool_module()
    data = {"work_schedule": "Hybrid"}
    tool.sync_remote_policy(data)
    assert data["remote_policy"] == "Hybrid"
    assert data["remote_percentage"] == 50


def test_sync_remote_policy_remote():
    tool = load_tool_module()
    data = {"work_schedule": "Remote"}
    tool.sync_remote_policy(data)
    assert data["remote_policy"] == "Remote"
    assert data["remote_percentage"] == 100


def test_sync_remote_policy_other():
    tool = load_tool_module()
    data = {"work_schedule": "Full-time"}
    tool.sync_remote_policy(data)
    assert data["remote_policy"] == "Onsite"
    assert "remote_percentage" not in data
