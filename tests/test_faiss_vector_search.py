import importlib.util
from pathlib import Path


def test_local_vector_store_search():
    path = Path(__file__).resolve().parents[1] / "services" / "vector_search.py"
    spec = importlib.util.spec_from_file_location("vector_search", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class DummyModel:
        def get_sentence_embedding_dimension(self):
            return 2

        def encode(self, texts):
            if isinstance(texts, list):
                return [[len(t), len(set(t))] for t in texts]
            return [[len(texts), len(set(texts))]]

    store = module.LocalVectorStore(texts=["foo", "bar"], model=DummyModel())
    result = store.search("foo", top_k=1)
    assert result == ["foo"]
