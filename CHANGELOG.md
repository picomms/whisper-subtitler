# Changelog

All notable changes to this project are documented in this file.

## 0.2.0 — 2026-07-23

Modernization release (Phases 0–5).

### Fixed

- Speaker labels from diarization now reach output formatters.

### Changed

- Package layout renamed to `whisper_subtitler` with a proper uv/setuptools install and console script.
- Transcription backend switched to **faster-whisper**; default model is **`large-v3`** with `--device` / `--compute-type` support (CPU `int8`, CUDA `float16` defaults).
- Diarization updated to **pyannote/speaker-diarization-3.1** with optional `num_speakers` / min/max bounds; custom clustering hacks removed.
- CLI slimmed to a focused surface; **JSON is the default primary output** (TXT/SRT/VTT/TTML still available via `-f`).
- Dev workflow standardized on **uv** + **Just** (+ tox / basedpyright).
- Documentation rewritten as the source of truth; historical Memory Bank/design notes moved to `docs/archive/`.

### Removed

- Legacy OpenAI Whisper entry path (`subwhisper.py`).
- ModelEvaluator / auto-model selection and related feature-creep CLI flags.
- Hand-rolled diarization clustering / preprocess CLI knobs.
