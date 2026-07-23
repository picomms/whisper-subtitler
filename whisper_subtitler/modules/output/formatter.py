"""
Output formatter main module.

This module provides the main OutputFormatter class that manages
the generation of different output formats.
"""

from pathlib import Path
from typing import Any

from ..logger import get_logger


class OutputFormatter:
    """Output formatter for transcription results.

    Manages the generation of various output formats such as
    TXT, SRT, VTT, and TTML from transcription results.
    """

    def __init__(self, config):
        """Initialize the formatter with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger("output")
        self.output_formats = config.output_formats
        self.output_dir = config.output_dir
        self.formatters = {}

        # Initialize formatters
        self._initialize_formatters()

    def _initialize_formatters(self):
        """Initialize the formatters for each supported output format."""

        from .formats import SRTFormatter, TTMLFormatter, TXTFormatter, VTTFormatter

        if "txt" in self.output_formats:
            self.formatters["txt"] = TXTFormatter(self.config)

        if "srt" in self.output_formats:
            self.formatters["srt"] = SRTFormatter(self.config)

        if "vtt" in self.output_formats:
            self.formatters["vtt"] = VTTFormatter(self.config)

        if "ttml" in self.output_formats:
            self.formatters["ttml"] = TTMLFormatter(self.config)

    def generate_outputs(self, result: dict[str, Any], base_name: str | None = None) -> dict[str, Path]:
        """Generate all configured output formats.

        Args:
            result: Transcription result with speaker labels
            base_name: Base name for output files (default: from input file)

        Returns:
            Dictionary of format to output file path
        """
        if not base_name:
            input_file = Path(self.config.input_file)
            base_name = input_file.stem

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_files = {}

        for fmt, formatter in self.formatters.items():
            try:
                output_path = output_dir / f"{base_name}.{fmt}"

                # Check if output already exists and force_overwrite is false
                if output_path.exists() and not self.config.force_overwrite:
                    self.logger.warning(f"Output file already exists, skipping: {output_path}")
                    output_files[fmt] = output_path
                    continue

                # Generate the output
                self.logger.info(f"Generating {fmt.upper()} file")
                formatter.generate(result, output_path)
                self.logger.info(f"{fmt.upper()} saved to: {output_path}")

                output_files[fmt] = output_path

            except Exception as e:
                self.logger.error(f"Failed to generate {fmt.upper()} file: {e!s}")

        return output_files
