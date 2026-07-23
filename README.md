# whisper-subtitler

[![Release](https://img.shields.io/github/v/release/picommcapp/whisper-subtitler)](https://img.shields.io/github/v/release/picommcapp/whisper-subtitler)
[![Build status](https://img.shields.io/github/actions/workflow/status/picommcapp/whisper-subtitler/main.yml?branch=main)](https://github.com/picommcapp/whisper-subtitler/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/picommcapp/whisper-subtitler/branch/main/graph/badge.svg)](https://codecov.io/gh/picommcapp/whisper-subtitler)
[![Commit activity](https://img.shields.io/github/commit-activity/m/picommcapp/whisper-subtitler)](https://img.shields.io/github/commit-activity/m/picommcapp/whisper-subtitler)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic subtitle generator with speaker identification using Whisper (via faster-whisper) and Pyannote. Accepts audio or video inputs (e.g. mp3, wav, m4a, flac, mp4, mkv); FFmpeg converts them to WAV for transcription.

- **Github repository**: <https://github.com/picommcapp/whisper-subtitler/>
- **Documentation** <https://picommcapp.github.io/whisper-subtitler/>

## Features

- Audio or video input via FFmpeg (mp3, wav, m4a, flac, mp4, mkv, and anything else FFmpeg can decode)
- Directory batch: pass a folder to process top-level media files sequentially
- Automatic transcription using Whisper via faster-whisper (CPU-friendly; CUDA optional)
- Speaker diarization using Pyannote `speaker-diarization-3.1`
- JSON-first output (TXT, SRT, VTT, TTML available on request)
- Configurable via .env file or command line
- GPU acceleration support when available

## Installation

Prerequisites: **Python ≥ 3.11**, **FFmpeg** on your `PATH`, and [uv](https://docs.astral.sh/uv/). [just](https://github.com/casey/just) is optional but used for local recipes.

1. Clone the repository
2. Install dependencies: `uv sync --all-groups` (or `just sync`)
3. Create a `.env` file based on `.env.sample`
4. Get a [HuggingFace token](https://hf.co/settings/tokens) and accept the [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) terms
5. Add the token to your `.env` file

See the [Installation guide](docs/installation.md) for details.

## Quick Usage

```bash
# Basic usage - writes JSON by default (model: large-v3)
uv run whisper-subtitler transcribe path/to/video.mp4
uv run whisper-subtitler transcribe path/to/talk.mp3

# Directory: process top-level media files sequentially (sorted by name)
uv run whisper-subtitler transcribe path/to/meetings/

# Explicit JSON (same as default)
uv run whisper-subtitler transcribe path/to/video.mp4 -f json

# Subtitle formats from the same segments
uv run whisper-subtitler transcribe path/to/video.mp4 -f srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f json srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f all

# Skip speaker identification
uv run whisper-subtitler transcribe path/to/video.mp4 --no-diarization

# Optional speaker bounds (meetings often leave these unset)
uv run whisper-subtitler transcribe path/to/video.mp4 -s 3
uv run whisper-subtitler transcribe path/to/video.mp4 --min-speakers 2 --max-speakers 6

# Specify output directory
uv run whisper-subtitler transcribe path/to/video.mp4 -o path/to/output

# Force CPU (default compute: int8)
uv run whisper-subtitler transcribe path/to/video.mp4 --device cpu

# CUDA uses float16 by default when available
uv run whisper-subtitler transcribe path/to/video.mp4 --device cuda

# Use a smaller model
uv run whisper-subtitler transcribe path/to/video.mp4 -m medium

# Or use the Just wrapper
just run transcribe path/to/video.mp4 --device cpu
just run transcribe path/to/talk.mp3 --device cpu
```

See [Usage](docs/usage.md) and the [FAQ](docs/faq.md) for more.

## Configuration

The application can be configured in three ways:

1. **Command Line Arguments**: Direct options when running the command
2. **Environment Variables**: Set in a `.env` file
3. **Configuration File**: Key/value file loaded with `--config`

### Environment Variables

Create a `.env` file in the project root with these options:

```bash
# Required for speaker identification (accept 3.1 model terms on HuggingFace)
HUGGINGFACE_TOKEN=your_token_here

# Whisper model configuration (faster-whisper)
WHISPER_MODEL_SIZE=large-v3   # tiny, base, small, medium, large-v3, turbo, etc.
WHISPER_LANGUAGE=en           # language code or empty for auto-detection
WHISPER_DEVICE=auto           # auto | cpu | cuda
WHISPER_COMPUTE_TYPE=         # optional: int8, int8_float16, int16, float16, float32

# Speaker diarization settings (pyannote/speaker-diarization-3.1)
SKIP_DIARIZATION=false      # set to true to disable speaker identification
NUM_SPEAKERS=               # exact count if known (takes precedence over min/max)
MIN_SPEAKERS=               # optional lower bound when count unknown
MAX_SPEAKERS=               # optional upper bound when count unknown

# Output settings (JSON is primary; add txt,srt,vtt,ttml as needed)
OUTPUT_FORMATS=json
```

For more options, see the `.env.sample` file.

## Development

Common development commands:

```bash
just sync
just check
just test
just docs
```

See the [Development guide](docs/development.md) and [CONTRIBUTING.md](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
