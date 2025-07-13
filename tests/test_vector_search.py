import asyncio
import types
import importlib.util
from pathlib import Path


def test_vector_search_returns_text(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "services" / "vector_search.py"
    spec = importlib.util.spec_from_file_location("vector_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyPaginator:
        def __aiter__(self):
            async def gen():
                yield types.SimpleNamespace(content=[types.SimpleNamespace(text="foo")])

            return gen()

    class DummyClient:
        def __init__(self) -> None:
            async def search(*args, **kwargs):
                return DummyPaginator()

            self.vector_stores = types.SimpleNamespace(search=search)

    vs = module.VectorStore(client=DummyClient(), store_id="id")
    result = asyncio.run(vs.search("bar"))
    assert result == ["foo"]
