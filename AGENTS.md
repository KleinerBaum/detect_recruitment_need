# AGENT Guidelines for Vacalyser

This document outlines how Codex agents should operate in this repository.

## Code requirements
- **Python** `>=3.10` with type hints.
- Follow **PEP 8** and ensure clean runs of `ruff`, `black` and `mypy`.
- Use Google‑style docstrings for all classes and functions.
- External dependencies are declared in `requirements.txt` and installed via `pip`.

## Project structure
- Entry point: `Recruitment_Need_Analysis_Tool.py`.

## Testing and static checks
Run the following before committing:
```bash
ruff .
black --check .
mypy .
pytest
```
All commands must succeed.

## Commit conventions
- Use feature branches (`feat/...`), open PRs against `dev`.
- Commit messages follow **Conventional Commits** (`feat:`, `fix:`, `chore:`, `docs:` …) with a short summary (max 60 characters).
- Provide migration scripts when models change.

## LLM‑specific guidelines
- Write unit tests with dummy returns before enabling real API calls.
- New prompts should integrate with existing templates in `utils/prompts.py`.

## Run the app
```bash
streamlit run `Recruitment_Need_Analysis_Tool.py`
```
