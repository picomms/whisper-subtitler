# Usage

The CLI entry point is `whisper-subtitler` (also available as `just run …` and the root `run.py` compatibility runner).

Input can be **audio or video**, or a **directory** of media files. Common examples: `mp3`, `wav`, `m4a`, `flac`, `mp4`, `mkv`. FFmpeg converts each file to a normalized WAV before transcription.

## Basic transcription

```bash
# Default: JSON output next to the input file (model: large-v3)
uv run whisper-subtitler transcribe path/to/video.mp4
uv run whisper-subtitler transcribe path/to/talk.mp3

# Directory: top-level media files only, processed sequentially in sorted order
uv run whisper-subtitler transcribe path/to/meetings/

# Explicit output directory
uv run whisper-subtitler transcribe path/to/video.mp4 -o path/to/output
```

When the input is a directory, outputs default to that directory. Non-media files and subdirectories are skipped. If one file fails, processing continues with the rest and the CLI exits non-zero.

## Progress feedback

On an interactive terminal you will see:

- Stage timings in the log (`Preparing audio…`, `Transcription done (12.3s)`, …)
- A faster-whisper progress bar during transcription
- A `Files: N/M` bar when processing a directory of media

Pipes and CI stay quiet (no bars). Use `-v` / `--verbose` for more detail and to force transcription progress even when stderr is not a TTY.

## Output formats

JSON is the primary structured output. Subtitle formats share the same segment list.

```bash
uv run whisper-subtitler transcribe path/to/video.mp4 -f json
uv run whisper-subtitler transcribe path/to/video.mp4 -f srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f json srt
uv run whisper-subtitler transcribe path/to/video.mp4 -f all   # json,txt,srt,vtt,ttml
```

### JSON shape

```json
{
  "language": "en",
  "text": "Full transcript text…",
  "segments": [
    {
      "start": 0.0,
      "end": 2.0,
      "text": "Hello",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

When diarization is skipped, segments use `"speaker": "Speaker"`.

## Diarization

```bash
# Skip speaker identification (no HuggingFace token required)
uv run whisper-subtitler transcribe path/to/video.mp4 --no-diarization

# Exact speaker count when known
uv run whisper-subtitler transcribe path/to/video.mp4 -s 3

# Bounds when the count is uncertain (common for meetings)
uv run whisper-subtitler transcribe path/to/video.mp4 --min-speakers 2 --max-speakers 6

# Override token from the CLI
uv run whisper-subtitler transcribe path/to/video.mp4 --token hf_…
```

If both `-s/--speakers` and min/max are set, the exact count wins and min/max are ignored.

## Models and device

```bash
uv run whisper-subtitler transcribe path/to/video.mp4 -m medium
uv run whisper-subtitler transcribe path/to/video.mp4 -m large-v3
uv run whisper-subtitler transcribe path/to/video.mp4 --device cpu
uv run whisper-subtitler transcribe path/to/video.mp4 --device cuda
uv run whisper-subtitler transcribe path/to/video.mp4 --compute-type int8
```

Useful models include `tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`, and `distil-large-v3`. Plain `large` aliases to `large-v3`.

Language:

```bash
uv run whisper-subtitler transcribe path/to/video.mp4 -l en
uv run whisper-subtitler transcribe path/to/video.mp4 -l auto
```

## Stable CLI surface

| Flag | Purpose |
|------|---------|
| `transcribe <input>` | File or directory of media files |
| `-o` / `--output` | Output directory |
| `-m` / `--model` | Whisper model size |
| `-l` / `--language` | Language code or `auto` |
| `-s` / `--speakers` | Exact speaker count |
| `--min-speakers` / `--max-speakers` | Speaker bounds |
| `-f` / `--format` | `json`, `txt`, `srt`, `vtt`, `ttml`, or `all` |
| `--no-diarization` | Transcription only |
| `--device` | `auto`, `cpu`, or `cuda` |
| `--compute-type` | faster-whisper compute type |
| `--cpu` | Alias for `--device cpu` |
| `--token` | HuggingFace token override |
| `--force` | Overwrite existing outputs |
| `--config` / `--env-file` | Config / `.env` paths |
| `-v` / `--log-level` / `--log-file` | Logging |
| `version` | Print package version |

## Environment variables

Common keys (see `.env.sample` for the full list):

| Variable | Notes |
|----------|--------|
| `HUGGINGFACE_TOKEN` | Required unless `--no-diarization` |
| `WHISPER_MODEL_SIZE` | Default `large-v3` |
| `WHISPER_LANGUAGE` | Language code or empty/`auto` |
| `WHISPER_DEVICE` | `auto` / `cpu` / `cuda` |
| `WHISPER_COMPUTE_TYPE` | Optional; defaults by device |
| `SKIP_DIARIZATION` | `true` to skip diarization |
| `NUM_SPEAKERS` / `MIN_SPEAKERS` / `MAX_SPEAKERS` | Speaker constraints |
| `OUTPUT_FORMATS` | Comma-separated; default `json` |

Whisper decoding knobs such as `WHISPER_BEAM_SIZE` remain available via env/config only (not first-class CLI flags).
