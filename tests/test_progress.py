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


def test_calc_extraction_progress():
    tool = load_tool_module()
    tool.ss.clear()
    tool.ss["extracted"] = {
        "BASIC": {"job_title": tool.ExtractResult("foo", 1.0)},
        "ROLE": {"task_list": tool.ExtractResult("bar", 1.0)},
    }
    pct = tool.calc_extraction_progress()
    assert pct > 0
