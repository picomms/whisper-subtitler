# whisper-subtitler

Automatic subtitle generation with speaker identification from audio or video.

- **Input:** audio or video via FFmpeg (e.g. `mp3`, `wav`, `m4a`, `flac`, `mp4`, `mkv`)
- **Transcription:** [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (default model `large-v3`)
- **Diarization:** [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1)
- **Primary output:** JSON (TXT, SRT, VTT, and TTML available on request)
- **Hardware:** CPU-first; CUDA optional

## Documentation

- [Installation](installation.md) — Python, FFmpeg, uv, HuggingFace token
- [Usage](usage.md) — CLI examples, formats, environment variables
- [FAQ](faq.md) — common issues and troubleshooting
- [Development](development.md) — Just/uv workflow and package layout
- [Modules](modules.md) — API reference (mkdocstrings)

## Quick start

```bash
# Install
uv sync --all-groups
cp .env.sample .env
# Edit .env: set HUGGINGFACE_TOKEN and accept the 3.1 model terms

# Transcribe (writes JSON by default)
uv run whisper-subtitler transcribe path/to/video.mp4
uv run whisper-subtitler transcribe path/to/talk.mp3

# Or via Just
just run transcribe path/to/video.mp4
```

See [Installation](installation.md) for prerequisites and [Usage](usage.md) for more examples.
