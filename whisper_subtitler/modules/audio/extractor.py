"""
Audio extraction module.

This module provides functionality to extract audio from
video files for transcription and diarization.
"""

import os
import subprocess
from pathlib import Path

from ..logger import get_logger


class AudioExtractor:
    """Audio extraction from video files.

    Handles extracting audio from video files using FFmpeg
    with configurable parameters for sample rate and channels.
    """

    def __init__(self, config):
        """Initialize the extractor with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger("audio")
        self.sample_rate = config.audio_sample_rate
        self.channels = config.audio_channels

    def extract_audio(self, input_file: str | Path, output_file: str | Path | None = None) -> Path:
        """Extract audio from a video file.

        Args:
            input_file: Path to the input video file
            output_file: Optional path for the output audio file

        Returns:
            Path to the extracted audio file

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

        self.logger.info(f"Extracting audio to: {output_path}")

        try:
            # Build FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                str(input_path),
                "-vn",  # Disable video
                "-acodec",
                "pcm_s16le",  # PCM 16-bit output
                "-ar",
                self.sample_rate,  # Sample rate
                "-ac",
                self.channels,  # Number of channels
                str(output_path),
            ]

            # Run FFmpeg command
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)

            return output_path

        except subprocess.CalledProcessError as e:
            error_message = e.stderr if hasattr(e, "stderr") else str(e)
            self.logger.error(f"Failed to extract audio: {error_message}")
            raise RuntimeError(f"Audio extraction failed: {error_message}")
        except Exception as e:
            self.logger.error(f"Unexpected error during audio extraction: {e!s}")
            raise
