"""
Transcription module.

This package handles audio transcription using
faster-whisper (CTranslate2 Whisper models).
"""

from .transcriber import SUPPORTED_MODELS, Transcriber, resolve_compute_type, resolve_device, resolve_model_name

__all__ = [
    "SUPPORTED_MODELS",
    "Transcriber",
    "resolve_compute_type",
    "resolve_device",
    "resolve_model_name",
]
