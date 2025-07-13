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
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_schema_contains_new_fields():
    tool = load_tool_module()
    assert "contract_end_date" in tool.KEY_TO_STEP
    assert "bonus_percentage" in tool.KEY_TO_STEP
