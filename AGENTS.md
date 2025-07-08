# Codex Agent Guidelines for Vacalyser

This guide describes how Codex agents should operate in this repository.

## Code requirements
- **Python** `>=3.10` with type hints.
- Frameworks: **Streamlit** for the UI, **Pydantic v2**, **OpenAI SDK** and **FAISS**.
- Follow **PEP 8** and ensure clean runs of `ruff`, `black` and `mypy`.
- Use Google‑style docstrings for all classes and functions.
- External dependencies are declared in `requirements.txt` and installed via `pip`.

## Project structure
- Streamlit entry point: `Recruitment_Need_Analysis_Tool.py`.
- Additional pages live in `pages/`.

## Testing and static checks
Run the following before committing:
```bash
ruff .
black --check .
mypy .
pytest
```
All commands must succeed.

## Git workflow
- Use feature branches (`feat/...`) and open PRs against `dev`.
- Commit messages follow **Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:` …) with a short summary (max 60 characters).
- Ensure lints and tests pass, update `README` or `CHANGELOG`, and provide migrations when models change.

## LLM‑specific guidelines
- Write unit tests with dummy returns before enabling real API calls.
- Use `services.vector_search.VectorStore.search()` to fetch concise context for prompts rather than sending whole documents.
- Validate agent responses with `models.VacancyProfile`. Log failures and trigger a correction prompt when parsing fails.
- Secrets must not be hard‑coded; read them via `os.getenv`.
- Prompt templates are in `utils/prompts.py`. Add language switches without overwriting existing templates.

## Run the app
```bash
streamlit run `Recruitment_Need_Analysis_Tool.py`
```
