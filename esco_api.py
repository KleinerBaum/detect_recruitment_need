"""Helper functions for the ESCO REST API."""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

BASE_URL = os.getenv("ESCO_API_BASE_URL", "https://ec.europa.eu/esco/api")


def _get_lang_entry(data: Any, language: str) -> str | None:
    """Return a localized string from a multilingual mapping."""

    if not isinstance(data, dict):
        return data if isinstance(data, str) else None

    entry = data.get(language)
    if entry is None:
        entry = data.get("en")
    if entry is None and data:
        entry = next(iter(data.values()))

    if isinstance(entry, dict):
        return entry.get("literal") or entry.get("title") or entry.get("label")
    if isinstance(entry, str):
        return entry
    return None


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
    relation: str = "hasEssentialSkill",
    language: str = "en",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return skills related to an occupation URI.

    By default uses the ``hasEssentialSkill`` relation which matches the
    ESCO API and retrieves skills essential for the given occupation.
    """
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


def fetch_occupation_details(uri: str, language: str = "en") -> dict[str, Any]:
    """Retrieve full occupation metadata from ESCO.

    Args:
        uri: Occupation URI.
        language: Preferred language for labels.

    Returns:
        Metadata dictionary including description and hierarchy.
    """
    params = {"uri": uri, "language": language, "full": "true"}
    try:
        resp = httpx.get(f"{BASE_URL}/resource/occupation", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if "description" in data:
            desc = _get_lang_entry(data["description"], language)
            if desc is not None:
                data["description"] = desc
        if "preferredLabel" in data:
            label = _get_lang_entry(data["preferredLabel"], language)
            if label is not None:
                data["preferredLabel"] = label
        return data
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO occupation details failed: %s", exc)
        return {}


def bulk_search_occupations(query: list[str]) -> dict[str, list[dict[str, Any]]]:
    """Search several job titles and return best matches."""
    results: dict[str, list[dict[str, Any]]] = {}
    for q in query:
        results[q] = search_occupations(q, limit=1)
    return results


def get_related_occupations(
    occupation_uri: str, language: str = "en", limit: int = 10
) -> list[dict[str, Any]]:
    """Suggest similar occupations using ESCO relations."""
    params = {
        "uri": occupation_uri,
        "relation": "isRelatedToOccupation",
        "language": language,
        "limit": str(limit),
    }
    try:
        resp = httpx.get(f"{BASE_URL}/resource/related", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("_embedded", {}).get("isRelatedToOccupation", [])
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO related occupations failed: %s", exc)
        return []


def get_skills_for_skill(
    skill_uri: str,
    *,
    relation: str = "isEssentialForSkill",
    language: str = "en",
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Fetch skills linked to a specific skill."""
    params = {
        "uri": skill_uri,
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
        logging.error("ESCO skill relation failed: %s", exc)
        return []


def get_skill_categories(skill_uri: str, language: str = "en") -> list[str]:
    """Retrieve broader categories for a skill."""
    params = {"uri": skill_uri, "language": language}
    try:
        resp = httpx.get(f"{BASE_URL}/resource/skill", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        groups = data.get("_embedded", {}).get("isGroupedBy", [])
        return [g.get("title") for g in groups if g.get("title")]
    except httpx.HTTPError as exc:  # pragma: no cover - log only
        logging.error("ESCO skill categories failed: %s", exc)
        return []


def get_occupation_statistics(occupation_uri: str) -> dict[str, Any]:
    """Aggregate statistics for an occupation."""
    details = fetch_occupation_details(occupation_uri)
    skills = get_skills_for_occupation(occupation_uri)
    return {
        "skills": len(skills),
        "languages": len(details.get("preferredLabel", {})),
    }
