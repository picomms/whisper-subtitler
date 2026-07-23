# Development

## Requirements

- Python ≥ 3.11
- [uv](https://docs.astral.sh/uv/)
- [just](https://github.com/casey/just)
- FFmpeg (for runtime audio extraction)

There is no pip requirements file or Makefile. Dependencies live in `pyproject.toml` / `uv.lock`; recipes live in the `Justfile`.

## Setup

```bash
just sync          # uv sync --all-groups
uv run pre-commit install
```

## Package layout

```text
whisper_subtitler/
  main.py
  version.py
  modules/
    application.py
    cli.py
    config.py
    audio/
    diarisation/
    output/
    transcribe/
tests/
docs/
Justfile
pyproject.toml
```

Import as `whisper_subtitler.modules…`. The console script is:

```text
whisper-subtitler = whisper_subtitler.modules.cli:main
```

## Just recipes

| Recipe | Purpose |
|--------|---------|
| `just sync` | Install all dependency groups |
| `just lock` | Refresh the lockfile |
| `just test` | Run pytest |
| `just lint` | Ruff check + format check |
| `just fmt` | Auto-fix with Ruff |
| `just typecheck` | basedpyright |
| `just check` | lint + typecheck + tests |
| `just run *args` | `uv run whisper-subtitler …` |
| `just docs` | `mkdocs build -s` |
| `just clean` | Remove caches/build artifacts |

## Tests and tox

```bash
just test
just check
uv run tox    # py311, py312, py313
```

## Docs

Published pages are under `docs/` (see `mkdocs.yml`). Historical Memory Bank / design notes live in `docs/archive/` and are not the source of truth.

```bash
just docs
```

## Version

The package version is defined once in `whisper_subtitler/version.py` and exposed via:

```bash
uv run whisper-subtitler version
```
