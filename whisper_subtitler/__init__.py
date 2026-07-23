"""
Whisper-Subtitler: Video transcription and diarization.

A tool for transcribing videos and identifying speakers
using OpenAI's Whisper and Pyannote.
"""

from .modules import Application, Config, get_logger, setup_logging
from .version import VERSION

__version__ = VERSION
__all__ = ["Application", "Config", "__version__", "get_logger", "setup_logging"]
