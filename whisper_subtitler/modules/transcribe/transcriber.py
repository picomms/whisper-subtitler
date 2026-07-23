"""
Audio transcription module using faster-whisper.

This module handles the transcription of audio files using
Whisper models via CTranslate2 (faster-whisper), with CPU-first
defaults and optional CUDA acceleration.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import torch
from faster_whisper import WhisperModel

from ..logger import get_logger

# Models supported by faster-whisper. Plain "large" maps to large-v3.
SUPPORTED_MODELS = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v3",
)


def resolve_model_name(model_size: str) -> str:
    """Map CLI/config model aliases to faster-whisper model ids."""
    if model_size == "large":
        return "large-v3"
    return model_size


def resolve_device(config) -> str:
    """Resolve inference device from config (cpu | cuda | auto / use_cuda)."""
    device = getattr(config, "device", None)
    if device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            return "cpu"
        return device

    # Legacy use_cuda / auto: prefer CUDA when available and not forced off
    use_cuda = getattr(config, "use_cuda", True)
    if use_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def resolve_compute_type(config, device: str) -> str:
    """Pick a compute type: explicit config, else best supported default for device.

    CUDA prefers float16 when supported. When it is not (e.g. Pascal GTX 1080),
    prefer int8 over float32 so large models fit in limited VRAM.
    """
    explicit = getattr(config, "compute_type", None)
    preferred = {
        "cuda": ["float16", "int8_float16", "int8", "int8_float32", "float32"],
        "cpu": ["int8", "int8_float32", "float32", "int16"],
    }.get(device, ["int8", "float32"])

    supported: set[str] | None = None
    try:
        import ctranslate2 as ct2

        supported = set(ct2.get_supported_compute_types(device))
    except Exception:
        supported = None

    # #region agent log
    try:
        import json as _json
        import time as _time

        with open("/home/cappy/code/whisper-subtitler/.cursor/debug-c34c50.log", "a") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "c34c50",
                        "runId": "post-fix",
                        "hypothesisId": "C",
                        "location": "transcriber.py:resolve_compute_type",
                        "message": "compute type resolution",
                        "data": {
                            "device": device,
                            "explicit": explicit,
                            "supported": sorted(supported) if supported is not None else None,
                            "chosen_preview": None,
                        },
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    def _first_supported() -> str:
        if supported is None:
            return preferred[0]
        for candidate in preferred:
            if candidate in supported:
                return candidate
        return next(iter(supported)) if supported else preferred[0]

    if explicit:
        if supported is not None and explicit not in supported:
            fallback = _first_supported()
            get_logger("transcribe").warning(
                f"compute_type={explicit} is not supported on {device} "
                f"(supported: {sorted(supported)}); using {fallback}"
            )
            return fallback
        return explicit

    chosen = _first_supported()
    if supported is not None and preferred[0] not in supported and chosen != preferred[0]:
        get_logger("transcribe").warning(
            f"Default {preferred[0]} is not supported on {device}; using {chosen} "
            f"(supported: {sorted(supported)})"
        )
    return chosen


def _is_cuda_oom_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def should_log_progress(config: Any) -> bool:
    """Show faster-whisper progress on interactive terminals or when verbose."""
    if getattr(config, "verbose", False):
        return True
    return sys.stderr.isatty()


class Transcriber:
    """Audio transcription using faster-whisper.

    Loads Whisper models via CTranslate2 and returns the legacy result shape
    so downstream diarization/formatters stay stable.
    """

    def __init__(self, config: Any):
        """Initialize the transcriber with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_size = config.model_size
        self.language = config.language
        self.logger = get_logger("transcribe")

        requested_device = getattr(config, "device", None)
        self.device = resolve_device(config)
        if requested_device == "cuda" and self.device == "cpu":
            self.logger.warning("CUDA was requested but is not available; falling back to CPU")
        self.compute_type = resolve_compute_type(config, self.device)
        self.model: WhisperModel | None = None
        self._cuda_oom_fell_back = False

        self.transcription_options: dict[str, Any] = {
            "language": self.language,
            "beam_size": getattr(config, "beam_size", 5),
            "best_of": getattr(config, "best_of", 5),
            "temperature": getattr(config, "temperature", 0.0),
            "initial_prompt": getattr(config, "initial_prompt", None),
        }
        self.transcription_options = {k: v for k, v in self.transcription_options.items() if v is not None}

    def _fallback_to_cpu_after_oom(self) -> None:
        """Drop the CUDA model and reload on CPU after an out-of-memory error."""
        self.logger.warning(
            f"CUDA out of memory with compute_type={self.compute_type}; falling back to CPU (int8)"
        )
        self.model = None
        self.device = "cpu"
        self.compute_type = "int8"
        self._cuda_oom_fell_back = True
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        # #region agent log
        try:
            import json as _json

            with open("/home/cappy/code/whisper-subtitler/.cursor/debug-c34c50.log", "a") as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "c34c50",
                            "runId": "post-fix",
                            "hypothesisId": "B",
                            "location": "transcriber.py:_fallback_to_cpu_after_oom",
                            "message": "fell back to CPU after OOM",
                            "data": {"device": self.device, "compute_type": self.compute_type},
                            "timestamp": int(time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

    def load_model(self) -> WhisperModel:
        """Load the faster-whisper model.

        Returns:
            Loaded WhisperModel
        """
        if self.model is None:
            model_name = resolve_model_name(self.model_size)
            self.logger.info(
                f"Loading faster-whisper model: {model_name} (device={self.device}, compute_type={self.compute_type})"
            )
            started = time.perf_counter()
            self.model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            elapsed = time.perf_counter() - started
            self.logger.info(f"Model loaded ({elapsed:.1f}s)")
        return self.model

    def _segments_to_result(self, segments: list[Any], info: Any) -> dict[str, Any]:
        """Convert faster-whisper segments/info into the legacy result shape."""
        result_segments: list[dict[str, Any]] = []
        texts: list[str] = []

        for i, segment in enumerate(segments):
            text = segment.text.strip()
            texts.append(text)
            result_segments.append({
                "id": getattr(segment, "id", i),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
                "speaker": None,
            })

        return {
            "text": " ".join(t for t in texts if t).strip(),
            "segments": result_segments,
            "language": getattr(info, "language", self.language),
        }

    def transcribe(self, audio_path: str, reference_text: str | None = None) -> dict[str, Any]:
        """Transcribe the given audio file.

        Args:
            audio_path: Path to the audio file
            reference_text: Unused; kept for call-site compatibility

        Returns:
            Dictionary containing transcription results in the legacy-compatible shape
        """
        del reference_text  # unused; retained for API compatibility
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        model = self.load_model()
        self.logger.info(f"Transcribing: {audio_path}")

        try:
            options = self.transcription_options.copy()
            options["log_progress"] = should_log_progress(self.config)
            self.logger.debug(f"Transcription options: {options}")
            # #region agent log
            try:
                import json as _json

                with open("/home/cappy/code/whisper-subtitler/.cursor/debug-c34c50.log", "a") as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "c34c50",
                                "runId": "post-fix",
                                "hypothesisId": "A",
                                "location": "transcriber.py:transcribe",
                                "message": "starting transcribe",
                                "data": {
                                    "device": self.device,
                                    "compute_type": self.compute_type,
                                    "audio": str(audio_path),
                                },
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            segments_gen, info = model.transcribe(str(audio_path), **options)
            segments = list(segments_gen)
            result = self._segments_to_result(segments, info)
            self.logger.info(
                f"Detected language '{result.get('language')}' "
                f"({getattr(info, 'language_probability', 0):.2f}), "
                f"{len(result['segments'])} segments"
            )
            return result
        except Exception as e:
            if self.device == "cuda" and not self._cuda_oom_fell_back and _is_cuda_oom_error(e):
                self.logger.error(f"Transcription error: {e!s}")
                self._fallback_to_cpu_after_oom()
                return self.transcribe(str(audio_path))
            self.logger.error(f"Transcription error: {e!s}")
            raise
