from __future__ import annotations

import os
from typing import List

from openai import AsyncOpenAI

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
