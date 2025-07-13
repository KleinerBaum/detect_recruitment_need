"""LangChain orchestration helpers."""

from __future__ import annotations
import logging


import httpx
from bs4 import BeautifulSoup
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING

_vs_spec = importlib.util.spec_from_file_location(
    "vector_search", Path(__file__).resolve().parent / "vector_search.py"
)
assert _vs_spec is not None
_vs_mod = importlib.util.module_from_spec(_vs_spec)
assert _vs_spec.loader is not None
_vs_spec.loader.exec_module(_vs_mod)
VectorStore = _vs_mod.VectorStore

if TYPE_CHECKING:  # pragma: no cover - typing only
    from services.vector_search import VectorStore as VectorStoreType


async def fetch_url_text(url: str) -> str:
    """Return cleaned text extracted from an URL."""
    try:
        resp = httpx.get(url, timeout=10)
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        logging.error("URL fetch failed: %s", exc)
        return ""
    soup = BeautifulSoup(resp.text, "html.parser")
    return soup.get_text(separator="\n").strip()


async def run_chain(
    url: str,
    question: str,
    *,
    vector_store: "VectorStoreType",
    client: ChatOpenAI | None = None,
    top_k: int = 5,
) -> str:
    """Execute the retrieval and LLM chain."""
    url_text = await fetch_url_text(url)
    snippets = await vector_store.search(question, top_k=top_k)
    llm = client or ChatOpenAI(model="gpt-4o", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Answer the user question based on the provided context."),
            (
                "human",
                "URL CONTENT:\n{url_text}\n\nOTHER CONTEXT:\n{context}\n\nQUESTION:\n{question}",
            ),
        ]
    )
    chain: Runnable[dict, str] = prompt | llm | StrOutputParser()
    return await chain.ainvoke(
        {
            "url_text": url_text[:5000],
            "context": "\n".join(snippets),
            "question": question,
        }
    )


__all__ = ["fetch_url_text", "run_chain"]
