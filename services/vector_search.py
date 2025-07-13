from __future__ import annotations

import os
from typing import List, Sequence

import faiss
import numpy as np
from openai import AsyncOpenAI
from sentence_transformers import SentenceTransformer

STORE_ID = os.getenv(
    "VACALYSER_VECTOR_STORE",
    "vs_67e40071e7608191a62ab06cacdcdd10",
)


class VectorStore:
    """Wrapper for OpenAI Vector Store search."""

    def __init__(
        self, client: AsyncOpenAI | None = None, store_id: str | None = None
    ) -> None:
        self.client = client or AsyncOpenAI()
        self.store_id = store_id or STORE_ID

    async def search(self, query: str, *, top_k: int = 5) -> List[str]:
        """Return snippet texts most relevant to the query."""
        paginator = await self.client.vector_stores.search(
            self.store_id,
            query=query,
            max_num_results=top_k,
        )
        results: List[str] = []
        async for item in paginator:
            text = " ".join(chunk.text for chunk in item.content if chunk.text)
            results.append(text)
        return results


class LocalVectorStore:
    """Simple FAISS-based vector store using SentenceTransformers."""

    def __init__(
        self,
        texts: Sequence[str] | None = None,
        *,
        model: SentenceTransformer | None = None,
    ) -> None:
        self.model = model or SentenceTransformer(
            os.getenv("VACALYSER_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        )
        self.texts: list[str] = list(texts or [])
        dim = self.model.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        if self.texts:
            self._add(self.texts)

    def _add(self, texts: Sequence[str]) -> None:
        vecs = self.model.encode(list(texts))
        self.index.add(np.asarray(vecs, dtype="float32"))

    def add(self, texts: Sequence[str]) -> None:
        """Add documents to the index."""

        self.texts.extend(texts)
        self._add(texts)

    def search(self, query: str, *, top_k: int = 5) -> list[str]:
        """Return texts with vectors closest to the query."""

        if self.index.ntotal == 0:
            return []
        vec = self.model.encode([query])
        distances, indices = self.index.search(np.asarray(vec, dtype="float32"), top_k)
        return [self.texts[i] for i in indices[0] if i < len(self.texts)]
