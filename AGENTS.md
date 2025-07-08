# Codex Agent Guidelines for Vacalyser

This guide describes how Codex agents should operate in this repository.

## Code requirements
- **Python** `>=3.10` with type hints.
- Frameworks: **Streamlit** for the UI, **Pydantic v2**, **OpenAI SDK** and **FAISS**.
- Follow **PEP 8**.
- External dependencies are declared in `requirements.txt`.

## Project structure
- Streamlit entry point: `Recruitment_Need_Analysis_Tool.py`.
- Additional pages live in `pages/`.

## Git workflow
- Use feature branches (`feat/...`) and open PRs against `dev`.
- Commit messages follow **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:` …) with a short summary (max 60 characters).

## LLM‑specific guidelines
- Write unit tests with dummy returns before enabling real API calls.
- Secrets must not be hard‑coded; read them via `os.getenv`.

## Run the app
```bash
streamlit run `Recruitment_Need_Analysis_Tool.py`
```
