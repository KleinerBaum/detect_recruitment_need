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


def test_conditional_role_fields_present():
    tool = load_tool_module()
    keys = {m["key"] for m in tool.SCHEMA["ROLE"]}
    assert {
        "travel_details",
        "on_call_expectations",
        "physical_duties_description",
    } <= keys
    assert tool.KEY_TO_STEP["travel_details"] == "ROLE"
