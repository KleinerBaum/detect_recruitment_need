import asyncio
import importlib.util
from pathlib import Path


def load_module():
    path = Path(__file__).resolve().parents[1] / "services" / "langchain_chain.py"
    spec = importlib.util.spec_from_file_location("chain", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore
    return module


class DummyVectorStore:
    called: str | None = None

    async def search(self, query: str, top_k: int = 5):
        DummyVectorStore.called = query
        return ["snippet"]


class DummyLLM:
    called: dict | None = None

    async def ainvoke(self, data):
        DummyLLM.called = data
        return "answer"

    def __call__(self, data):
        return asyncio.run(self.ainvoke(data))


def test_run_chain(monkeypatch):
    mod = load_module()

    def dummy_get(url, timeout=10):
        class Resp:
            text = "<p>hello</p>"

            def raise_for_status(self) -> None:
                pass

        return Resp()

    monkeypatch.setattr(mod.httpx, "get", dummy_get)
    vs = DummyVectorStore()
    llm = DummyLLM()
    out = asyncio.run(mod.run_chain("http://x", "q?", vector_store=vs, client=llm))
    assert out == "answer"
    assert DummyVectorStore.called == "q?"
    assert "hello" in DummyLLM.called.messages[1].content
