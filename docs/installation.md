# Installation

## Prerequisites

- **Python** ≥ 3.11
- **[uv](https://docs.astral.sh/uv/)** for dependency management
- **[FFmpeg](https://ffmpeg.org/)** on your `PATH` (required to decode audio/video inputs such as mp3, wav, m4a, flac, mp4, mkv into normalized WAV)
- **[just](https://github.com/casey/just)** (optional; used for local/CI recipes)
- A **HuggingFace access token** if you use speaker diarization

Optional: a CUDA-capable GPU and matching PyTorch/CUDA stack for faster inference.

## Install the project

```bash
git clone https://github.com/picommcapp/whisper-subtitler.git
cd whisper-subtitler
uv sync --all-groups
```

Or with Just:

```bash
just sync
```

## Configure environment

```bash
cp .env.sample .env
```

Edit `.env`:

1. Set `HUGGINGFACE_TOKEN` to a token from [HuggingFace settings](https://hf.co/settings/tokens).
2. Accept the user conditions for [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) (and any gated model cards it requires).

If you only need transcription, you can skip the token and always pass `--no-diarization`.

## Verify

```bash
uv run whisper-subtitler version
# or
just run version
```

## Device notes

- Default device is `auto`: CUDA when available, otherwise CPU.
- Force CPU with `--device cpu` (CPU default compute type is `int8`).
- CUDA default compute type is `float16` when a GPU is selected.
