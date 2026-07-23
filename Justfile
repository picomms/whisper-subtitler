set shell := ["bash", "-uc"]

help:
    @just --list

sync:
    uv sync --all-groups

lock:
    uv lock

test:
    uv run python -m pytest tests

lint:
    uv run ruff check .
    uv run ruff format --check .

fmt:
    uv run ruff check --fix .
    uv run ruff format .

typecheck:
    uv run basedpyright whisper_subtitler tests

check: lint typecheck test

run *args:
    uv run whisper-subtitler {{args}}

docs:
    uv run mkdocs build -s

clean:
    rm -rf .pytest_cache .ruff_cache .mypy_cache .pyright dist build *.egg-info pytest-cache-files-*
    shopt -s globstar nullglob; rm -rf **/__pycache__
