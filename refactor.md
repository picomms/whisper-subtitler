# Refactor / Modernization Plan

**Status:** Decisions locked — planning only, no implementation yet  
**Date:** 2026-07-23 (updated with stakeholder answers)  
**Goal:** Bring whisper-subtitler up to date while staying a focused CLI tool (transcription + diarization → subtitle files). Avoid feature creep.

---

## 1. Current state (brief)

| Area | Today | Problem |
|------|--------|---------|
| Transcription | Legacy Whisper backend, models: `tiny`…`large` | No `large-v2` / `large-v3` / `turbo` in CLI or config |
| Diarization | `pyannote/speaker-diarization` (legacy) | Current pipelines are `3.1` and `community-1`; API/token args changed |
| Packaging | `pyproject.toml` + `uv.lock`; CI uses `uv` | README/docs still say `pip install -r requirements.txt` (file does not exist) |
| Dev UX | CONTRIBUTING references `make check` / `make test` | No Makefile; no Justfile |
| Package layout | Directory named `whisper-subtitler/` (hyphen) | Awkward as an importable package; relies on `run.py` path hacks |
| Version | `pyproject.toml` = `0.0.1`, `version.py` = `0.1.0` | Drift |

**Critical bug:** In `application.py`, speaker assignment builds `combined_result` but `generate_outputs()` is still called with the original `transcription` — so diarized speaker labels never reach outputs. Fix early, independent of other upgrades.

---

## 2. Locked decisions

| # | Topic | Decision |
|---|--------|----------|
| 1 | Transcription engine | **`faster-whisper`** (CPU-first; locked after stakeholder follow-up) |
| 2 | Default model | **`large-v3`** |
| 3 | Diarization pipeline | **`pyannote/speaker-diarization-3.1`** |
| 4 | CLI / outputs | **Focused slim-down**; **JSON first**, then other formats |
| 5 | Package rename | **Yes** → `whisper_subtitler/` |
| 6 | Hardware | **CPU first-class**; CUDA optional (`--device cpu\|cuda\|auto`) |
| 7 | Python | **`>=3.11`** |
| 8 | Domain | Council/meeting videos; **speaker count often unknown** — keep optional `--speakers` / min/max, default to auto |
| 9 | Task runner | **Just** (Justfile), not Make |
| 10 | Docs | **Prune** stale Memory Bank; rewrite as source of truth **after** refactor |

---

## 3. Whisper: large-v3 — do we need to change engines?

### Answer to “Why change?”

**You do not need to switch engines to get `large-v3`.**

The previous Whisper backend already supported `large-v3` (and `turbo`). The original gap was that our CLI/config only exposed `tiny`…`large`. Wiring `large-v3` as default was a small change on that stack.

**Why people switch to `faster-whisper` anyway (optional later):**

| | Previous backend | `faster-whisper` |
|--|---------------------------|------------------|
| Model family | Same Whisper weights (`large-v3`) | Same |
| Speed | Baseline | Often ~4× faster |
| Memory | Higher | Lower (CTranslate2; int8/float16) |
| CPU runs | Workable but slow on long meetings | Much more practical on CPU |
| Change cost | None | Adapter for result shape + new dep |

Given **CPU is first-class** and council videos can be long, `faster-whisper` would help day-to-day — but it is an optimization, not a prerequisite for “full v3.”

**Locked for this refactor:** switch to **`faster-whisper`**, default **`large-v3`**, CPU-first (`int8` on CPU, `float16` on CUDA).

Support end-to-end: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, **`large-v3`**, optionally `turbo`.

---

## 4. Diarization — locked to 3.1

Replace legacy `pyannote/speaker-diarization` with:

```python
Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    token=hf_token,
)
# pipeline(audio, num_speakers=…)  # when known
# pipeline(audio, min_speakers=…, max_speakers=…)  # when bounded
# pipeline(audio)  # when unknown (common for meetings)
```

Update HF license URLs in docs / `.env.sample` / errors to the 3.1 model card.

### Simplify custom clustering

Remove hand-rolled clustering / temporal merge / speaker-count hacks. Rely on pipeline `num_speakers` / `min_speakers` / `max_speakers`. Less code, fewer silent failures. Speaker count remains **optional** because meetings often do not have a known count.

---

## 5. Outputs — JSON first, then the rest

**Priority order after slim-down:**

1. **JSON** — primary structured output (segments with `start`, `end`, `text`, `speaker`, plus metadata). Useful for downstream tooling and as source of truth for tests.
2. **SRT** / **VTT** / **TTML** / **TXT** — restore/keep after JSON is solid, or generate from the same internal segment list.

Slim format CLI, e.g. `-f json` default (or `json` + one subtitle format). Avoid “always write all four formats.”

---

## 6. CLI slim-down (focused)

**Stable surface (keep):**

- `transcribe <input>`
- `-o` / `--output`
- `-m` / `--model` (incl. `large-v3`)
- `-l` / `--language` (default sensible for meetings; allow auto)
- `-s` / `--speakers`, `--min-speakers`, `--max-speakers` (all optional)
- `-f` / `--format` (JSON first; others as they land)
- `--no-diarization`
- `--device cpu|cuda|auto` (default `auto`: CUDA if available, else CPU)
- `--token` / env for HF
- `-v` / `--log-level` / `--log-file`
- `version`

**Remove or defer (feature creep):**

- Auto model selection / model criteria / `ModelEvaluator`
- Optimize-for-audio tournament
- Cluster / optimize-speakers / preprocess flags
- Per-flag noise reduction / high-pass / normalize toggles (use sensible fixed preprocessing or none)
- Beam/best-of/temperature/initial-prompt as first-class CLI (env/config only if needed)

---

## 7. Tooling: uv + Just + package rename

### uv

1. `pyproject.toml` + `uv.lock` = source of truth (no `requirements.txt`)
2. Docs → `uv sync` / `uv run`
3. `[project.scripts]` entry point
4. `requires-python = ">=3.11"`
5. CI stays on `uv sync --frozen` / `--locked`

### Package layout

Rename `whisper-subtitler/` → **`whisper_subtitler/`**. Thin `run.py` optional during transition; prefer entry point.

### Justfile recipes

| Recipe | Purpose |
|--------|---------|
| `just sync` | `uv sync --all-groups` |
| `just lock` | `uv lock` |
| `just test` | `uv run pytest` |
| `just lint` | ruff check + format check |
| `just fmt` | ruff format |
| `just typecheck` | one tool only (prefer basedpyright *or* mypy — pick during implement) |
| `just check` | lint + typecheck + tests |
| `just run *args` | invoke CLI via uv |
| `just docs` | mkdocs serve/build |
| `just clean` | caches / build / temp artifacts |

---

## 8. Code review priorities (still valid)

### Must-fix

1. Speaker labels discarded → pass combined result into formatters; regression test (JSON + later SRT).
2. Model choices include `large-v3`; default `large-v3`.
3. Docs/tooling mismatch (pip, Make, stale Memory Bank).
4. Single version source.

### Should-fix

5. Slim CLI as in §6.
6. Remove `ModelEvaluator` / auto-model from user path.
7. Delete or archive legacy `subwhisper.py`.
8. Revive stubbed tests; JSON speaker-label regression.
9. Align defaults (config vs README).
10. Fix `python-dotenv` dependency declaration if wrong.
11. One typechecker in CI + Just.
12. FFmpeg required; fail fast with clear message.
13. Device: CPU default path must work; CUDA when present and selected.

---

## 9. Phased plan (revised)

| Phase | Work | Outcome |
|-------|------|---------|
| **0** | **Done:** Fix speaker-label bug + test | Diarization reaches formatters |
| **1** | **Done:** Package rename → `whisper_subtitler`; uv entry point; Justfile; Python ≥3.11 | Installable, modern workflow |
| **2** | **Done:** `faster-whisper` + **`large-v3` default**; model name plumbing; `--device`; compute validation | Modern transcription, CPU/CUDA |
| **3** | **Done:** pyannote **`speaker-diarization-3.1`**; drop custom clustering; optional speaker bounds | Modern diarization |
| **4** | **Done:** Slim CLI; **JSON output first**; then SRT/VTT/TTML/TXT from same segments | Focused product surface |
| **5** | **Done:** Prune stale `docs/`; rewrite README + user docs as source of truth; version bump | Thaw complete |

Optional later: ~~**2b** — trial `faster-whisper`~~ **Done** — `faster-whisper` is the transcription backend.

Each phase should stay mergeable alone.

---

## 10. Docs after refactor

- **Prune / archive:** Memory Bank-style files that disagree with reality (`progress.md`, old creative-phase docs, outdated GPU expansion plans, etc.) — either delete or move to `docs/archive/` with a short note.
- **Keep / rewrite as SoT:** README, install, usage, CLI reference, HF token + 3.1 model acceptance, Just + uv workflow.
- Do this in **Phase 5**, after code is true — not speculative docs mid-flight.

---

## 11. Decision log (closed)

| Decision | Locked |
|----------|--------|
| Transcription backend | `faster-whisper` |
| Default model | `large-v3` |
| Diarization model | `pyannote/speaker-diarization-3.1` |
| Custom clustering | Remove; use pipeline params |
| Primary output | JSON first |
| CLI | Focused slim-down |
| Package dir rename | Yes → `whisper_subtitler` |
| Hardware | CPU first-class; CUDA optional |
| Python | `>=3.11` |
| Speakers | Optional; auto when unknown |
| Task runner | Just |
| Docs | Prune + rewrite after refactor |

---

*Thaw complete (Phases 0–5). Maintenance / incremental improvements from here.*
