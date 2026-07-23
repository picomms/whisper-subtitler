"""
Application orchestrator module.

This module provides the main application class that orchestrates
the entire transcription and diarization process.
"""

from __future__ import annotations

import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from tqdm import tqdm

from .audio import AudioExtractor
from .config import Config
from .diarisation import Diarizer
from .logger import setup_logging
from .output import OutputFormatter
from .transcribe import Transcriber

MEDIA_EXTENSIONS = frozenset({
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".opus",
    ".aac",
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".avi",
})


def resolve_input_files(path: Path) -> list[Path]:
    """Resolve a file or directory path to a list of media inputs.

    - Missing path raises FileNotFoundError.
    - A regular file is returned as a single-item list (any extension).
    - A directory expands to top-level media files only (non-recursive),
      sorted by name. Empty media set raises ValueError.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    if path.is_file():
        return [path]

    if path.is_dir():
        media_files = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)
        if not media_files:
            raise ValueError(
                f"No media files found in directory: {path} "
                f"(supported extensions: {', '.join(sorted(ext.lstrip('.') for ext in MEDIA_EXTENSIONS))})"
            )
        return media_files

    raise ValueError(f"Input path is neither a file nor a directory: {path}")


@contextmanager
def timed_stage(logger: Any, name: str) -> Iterator[None]:
    """Log stage start and finish with elapsed seconds."""
    logger.info(f"{name}…")
    started = time.perf_counter()
    try:
        yield
    except Exception:
        elapsed = time.perf_counter() - started
        logger.error(f"{name} failed ({elapsed:.1f}s)")
        raise
    else:
        elapsed = time.perf_counter() - started
        logger.info(f"{name} done ({elapsed:.1f}s)")


class Application:
    """Main application orchestrator.

    Coordinates all components of the whisper-subtitler
    pipeline including configuration, audio extraction,
    transcription, diarization, and output generation.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the application with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config or Config()

        # Configure logging
        self.logger = setup_logging(self.config)

        # Initialize components to None (will be lazily loaded)
        self.audio_extractor = None
        self.transcriber = None
        self.diarizer = None
        self.output_formatter = None

    def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing application components")

        # Initialize components
        self.audio_extractor = AudioExtractor(self.config)
        self.transcriber = Transcriber(self.config)
        if not self.config.skip_diarization:
            self.diarizer = Diarizer(self.config)
        self.output_formatter = OutputFormatter(self.config)

        self.logger.debug("All components initialized")

    def process_file(self, input_file: str | Path) -> dict[str, Any]:
        """Process a single media file through the full pipeline.

        Args:
            input_file: Path to the input media file

        Returns:
            Dictionary of output paths by format

        Raises:
            Exception: If any part of the process fails
        """
        if not self.audio_extractor or not self.transcriber or not self.output_formatter:
            self.initialize()

        input_path = Path(input_file)
        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")
        if not input_path.is_file():
            raise ValueError(f"Input path is not a file: {input_path}")

        # Keep config/formatter in sync with the current file
        self.config.input_file = str(input_path)

        # Determine output directory
        output_dir = Path(self.config.output_dir) if self.config.output_dir else input_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        base_name = input_path.stem
        audio_file = output_dir / f"{base_name}.wav"
        same_path = input_path.resolve() == audio_file.resolve()

        # Always (re)convert when targeting the same WAV path so sample rate/channels
        # match config; otherwise extract when missing or --force.
        with timed_stage(self.logger, "Preparing audio"):
            if same_path or not audio_file.exists() or self.config.force_overwrite:
                self.logger.info(f"Preparing audio at: {audio_file}")
                self.audio_extractor.extract_audio(input_file=str(input_path), output_file=str(audio_file))
            else:
                self.logger.info(f"Using existing audio file: {audio_file}")

        with timed_stage(self.logger, "Transcription"):
            self.logger.debug(f"Using model size: {self.config.model_size}")
            transcription = self.transcriber.transcribe(str(audio_file))

        if self.config.skip_diarization:
            self.logger.info("Skipping speaker diarization as requested")
            for segment in transcription["segments"]:
                segment["speaker"] = "Speaker"
        else:
            with timed_stage(self.logger, "Diarization"):
                self.logger.info("Initializing speaker diarization pipeline")
                self.diarizer.initialize_pipeline()

                self.logger.info(f"Running speaker diarization on {audio_file}")
                speaker_segments = self.diarizer.diarize(str(audio_file))

                self.logger.info("Combining transcription and diarization results")
                self.logger.debug(f"Found {len(speaker_segments)} speaker segments")
                self.logger.debug(f"Transcription has {len(transcription['segments'])} segments")
                self.logger.info("Assigning speakers to transcription segments")
                transcription = self.diarizer.assign_speakers_to_segments(transcription, speaker_segments)

        with timed_stage(self.logger, "Outputs"):
            output_files = self.output_formatter.generate_outputs(transcription, base_name)

        self.logger.info(f"Processing complete for {input_path}")
        return output_files

    def process(self) -> dict[str, dict[str, Any]]:
        """Process config.input_file (a single file or a directory of media).

        Files are processed sequentially. Per-file failures are collected and
        processing continues with the remaining inputs.

        Returns:
            Dictionary with ``results`` (input path → output paths by format)
            and ``failures`` (input path → error message).

        Raises:
            FileNotFoundError: If the input path does not exist
            ValueError: If a directory contains no media files
        """
        if not self.audio_extractor or not self.transcriber or not self.output_formatter:
            self.initialize()

        input_path = Path(self.config.input_file)
        files = resolve_input_files(input_path)

        results: dict[str, dict[str, Any]] = {}
        failures: dict[str, str] = {}

        self.logger.info(f"Processing {len(files)} file(s) sequentially")

        # Multi-file batch: show N/M progress; single file relies on Whisper's bar
        if len(files) > 1:
            file_iter = tqdm(
                files,
                desc="Files",
                unit="file",
                disable=not sys.stderr.isatty(),
            )
        else:
            file_iter = files

        for file_path in file_iter:
            key = str(file_path)
            try:
                self.logger.info(f"Processing {file_path}")
                results[key] = self.process_file(file_path)
            except Exception as e:
                self.logger.error(f"Processing failed for {file_path}: {e!s}")
                failures[key] = str(e)

        return {"results": results, "failures": failures}
