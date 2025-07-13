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


def test_estimate_salary_range_senior_ds():
    tool = load_tool_module()
    assert tool.estimate_salary_range("Data Scientist", "Senior") == "86400–105600 €"


def test_calculate_total_compensation():
    tool = load_tool_module()
    total = tool.calculate_total_compensation(
        (50000, 60000), ["Company Car", "Health Insurance"]
    )
    assert total == 65500


def test_predict_annual_salary_breakdown():
    tool = load_tool_module()
    salary, parts = tool.predict_annual_salary(
        "Engineer",
        "desc words",
        "tasks list",
        "Berlin",
        ["Python", "SQL"],
    )
    assert salary == 30000 + sum(parts.values())
    assert parts["skills"] == 2000


def test_salary_range_parser_and_warning():
    tool = load_tool_module()
    assert tool.parse_salary_range("50000-60000") == (50000, 60000)
    assert tool.salary_deviation_high((30000, 40000), (50000, 60000))
    assert not tool.salary_deviation_high((51000, 61000), (50000, 60000))
