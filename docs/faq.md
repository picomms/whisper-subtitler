# FAQ

## HuggingFace token / model access errors

Diarization needs a HuggingFace token and acceptance of the [pyannote/speaker-diarization-3.1](https://hf.co/pyannote/speaker-diarization-3.1) terms (and related gated models).

- Set `HUGGINGFACE_TOKEN` in `.env`, or pass `--token`.
- Or skip diarization: `--no-diarization` / `SKIP_DIARIZATION=true`.

## FFmpeg not found

Audio and video inputs are converted with FFmpeg (for example MP3 → WAV). Install it system-wide and ensure `ffmpeg` is on your `PATH`.

## Does MP3 (or other audio) work?

Yes. Pass any audio or video file FFmpeg can decode (`mp3`, `wav`, `m4a`, `flac`, `mp4`, `mkv`, …). The tool converts it to a normalized WAV, then transcribes (and optionally diarizes) that WAV. FFmpeg must be installed.

## Can I process a whole directory?

Yes. Pass a directory path to `transcribe`. Top-level media files (by extension) are processed **sequentially** in sorted filename order — not recursively, and not in parallel. Non-media files and subdirectories are skipped. If one file fails, the rest still run; the CLI exits non-zero when any failure occurred.

## Is the process frozen? How do I see progress?

Long meetings spend most time in transcription and diarization. On an interactive terminal you get stage timings (with elapsed seconds), a Whisper progress bar during transcription, and a file-level bar for directory batches. Redirected output (pipes/CI) disables bars; use `-v` for more stage detail.

## CUDA was requested but fell back to CPU

With `--device cuda` (or `auto` when CUDA is expected), the app falls back to CPU if `torch.cuda.is_available()` is false and logs a warning. Use `--device cpu` explicitly when you intend CPU-only runs.

## CPU is slow / out of memory on GPU

- CPU default compute type is `int8`; CUDA default is `float16`.
- Override with `--compute-type` (for example `int8`, `float16`, `float32`).
- Try a smaller model (`-m medium` or `-m small`) for long meetings.

## Speakers look wrong or missing

- Confirm diarization is enabled (not `--no-diarization`) and the HF token is valid.
- Provide `-s` when the speaker count is known, or `--min-speakers` / `--max-speakers` when you only know a range.
- Meeting recordings with heavy overlap remain hard; JSON segments are the best place to inspect labels.

## I want subtitles, not only JSON

JSON is the default. Request subtitle formats explicitly:

```bash
uv run whisper-subtitler transcribe video.mp4 -f srt
uv run whisper-subtitler transcribe video.mp4 -f json srt
uv run whisper-subtitler transcribe video.mp4 -f all
```

## Removed CLI flags

Flags such as `--auto-model`, `--cluster`, `--preprocess`, and per-flag audio toggles were removed in the Phase 4 slim-down. Prefer the stable surface documented in [Usage](usage.md).
