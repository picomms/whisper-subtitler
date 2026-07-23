"""
Audio extraction module.

This module provides functionality to prepare audio from
audio or video media files for transcription and diarization.
"""

import os
import subprocess
from pathlib import Path
from typing import Any

from ..logger import get_logger


class AudioExtractor:
    """Audio preparation from audio or video media files.

    Uses FFmpeg to decode media and write PCM WAV with configurable
    sample rate and channels (suitable for Whisper and pyannote).
    """

    def __init__(self, config: Any):
        """Initialize the extractor with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger("audio")
        self.sample_rate = config.audio_sample_rate
        self.channels = config.audio_channels

    def extract_audio(self, input_file: str | Path, output_file: str | Path | None = None) -> Path:
        """Prepare audio from an audio or video media file.

        Args:
            input_file: Path to the input media file (e.g. mp3, wav, mp4)
            output_file: Optional path for the output WAV file

        Returns:
            Path to the prepared WAV audio file

        Raises:
            FileNotFoundError: If the input file doesn't exist
            RuntimeError: If FFmpeg extraction fails
        """
        # Resolve paths
        input_path = Path(input_file).resolve()

        if not input_path.exists():
            self.logger.error(f"Input file not found: {input_path}")
            raise FileNotFoundError(f"Input file not found: {input_path}")

        # Generate output path if not provided
        if output_file is None:
            # Use same directory as input file
            output_path = input_path.with_suffix(".wav")
        else:
            output_path = Path(output_file).resolve()

        # Create output directory if it doesn't exist
        os.makedirs(output_path.parent, exist_ok=True)

        # Avoid FFmpeg reading and writing the same path in-place
        same_path = input_path == output_path
        ffmpeg_output = output_path
        if same_path:
            ffmpeg_output = output_path.with_name(f"{output_path.stem}.converted{output_path.suffix}")

        self.logger.info(f"Preparing audio from {input_path} → {output_path}")

        try:
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                str(input_path),
                "-vn",  # Disable video (no-op for audio-only inputs)
                "-acodec",
                "pcm_s16le",  # PCM 16-bit output
                "-ar",
                self.sample_rate,  # Sample rate
                "-ac",
                self.channels,  # Number of channels
                str(ffmpeg_output),
            ]

            # Run FFmpeg command
            subprocess.run(cmd, check=True, capture_output=True, text=True)

            if same_path:
                os.replace(ffmpeg_output, output_path)

            return output_path

        except subprocess.CalledProcessError as e:
            error_message = e.stderr if hasattr(e, "stderr") else str(e)
            self.logger.error(f"Failed to prepare audio: {error_message}")
            raise RuntimeError(f"Audio extraction failed: {error_message}")
        except Exception as e:
            self.logger.error(f"Unexpected error during audio preparation: {e!s}")
            raise
