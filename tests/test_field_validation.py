import importlib.util
from pathlib import Path
import sys


def load_validation_module():
    path = Path(__file__).resolve().parents[1] / "validation_utils.py"
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location("validation_utils", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


def test_parse_date_value_valid():
    mod = load_validation_module()
    assert mod.parse_date_value("2025-09-01") == "2025-09-01"


def test_parse_date_value_invalid():
    mod = load_validation_module()
    assert mod.parse_date_value("not a date") is None


def test_validate_email():
    mod = load_validation_module()
    assert mod.validate_email("user@example.com") == "user@example.com"
    assert mod.validate_email("bad@email") is None


def test_validate_phone():
    mod = load_validation_module()
    assert mod.validate_phone("+123-456-789") == "+123-456-789"
    assert mod.validate_phone("phone123") is None
