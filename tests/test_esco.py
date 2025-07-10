import importlib.util
import sys
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "esco_api.py"
    spec = importlib.util.spec_from_file_location("esco_api", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader
    spec.loader.exec_module(module)
    return module


class DummyResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self) -> None:  # pragma: no cover - dummy
        pass

    def json(self):
        return self._data


class DummyClient:
    def __init__(self, data):
        self.data = data

    def __call__(self, *args, **kwargs):
        return self

    def get(self, url, params=None, timeout=10):
        return DummyResponse(self.data)


def test_search_occupations(monkeypatch):
    mod = load_module()
    data = {"_embedded": {"results": [{"label": "Nurse"}]}}
    monkeypatch.setattr(mod.httpx, "get", DummyClient(data).get)
    res = mod.search_occupations("nurse")
    assert res == data["_embedded"]["results"]


def test_get_skills_for_occupation(monkeypatch):
    mod = load_module()
    relation = "isEssentialForSkill"
    data = {"_embedded": {relation: [{"uri": "skill1"}]}}
    monkeypatch.setattr(mod.httpx, "get", DummyClient(data).get)
    res = mod.get_skills_for_occupation("occ", relation=relation)
    assert res == data["_embedded"][relation]


def test_suggest(monkeypatch):
    mod = load_module()
    data = {"_embedded": {"results": [{"label": "abc"}]}}
    monkeypatch.setattr(mod.httpx, "get", DummyClient(data).get)
    res = mod.suggest("a", type_="skill")
    assert res == data["_embedded"]["results"]
