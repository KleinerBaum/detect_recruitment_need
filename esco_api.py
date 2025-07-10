"""Helper functions for the ESCO REST API."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

BASE_URL = os.getenv("ESCO_API_BASE_URL", "https://ec.europa.eu/esco/api")


def search_occupations(
    text: str, language: str = "en", limit: int = 10
) -> list[dict[str, Any]]:
    """Return occupation matches for the given text."""
    params = {
        "text": text,
        "language": language,
        "type": "occupation",
        "limit": str(limit),
        "full": "false",
    }
    try:
        resp = httpx.get(f"{BASE_URL}/search", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("_embedded", {}).get("results", [])
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO search failed: %s", exc)
        return []


def get_skills_for_occupation(
    occupation_uri: str,
    *,
    relation: str = "isEssentialForSkill",
    language: str = "en",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return skills related to an occupation URI."""
    params = {
        "uri": occupation_uri,
        "relation": relation,
        "language": language,
        "limit": str(limit),
    }
    try:
        resp = httpx.get(f"{BASE_URL}/resource/related", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("_embedded", {}).get(relation, [])
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO related lookup failed: %s", exc)
        return []


def suggest(
    text: str, *, type_: str, language: str = "en", limit: int = 10
) -> list[dict[str, Any]]:
    """Return autocomplete suggestions for skills or occupations."""
    params = {
        "text": text,
        "language": language,
        "type": type_,
        "limit": str(limit),
    }
    try:
        resp = httpx.get(f"{BASE_URL}/suggest2", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("_embedded", {}).get("results", [])
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO suggest failed: %s", exc)
        return []
