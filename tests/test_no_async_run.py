import re
from pathlib import Path


def test_no_asyncio_run_in_callbacks():
    path = Path(__file__).resolve().parents[1] / "Recruitment_Need_Analysis_Tool.py"
    text = path.read_text()
    runs = re.findall(r"asyncio\.run\(", text)
    assert len(runs) == 1
