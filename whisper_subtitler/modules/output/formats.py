"""
Output formatters for different subtitle and transcript formats.

This module provides formatters for various output formats,
including TXT, SRT, VTT, and TTML.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape as xml_escape

from tqdm import tqdm

from ..logger import get_logger


class BaseFormatter(ABC):
    """Base class for all output formatters."""

    def __init__(self, config):
        """Initialize the formatter with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.logger = get_logger(f"output.{self.__class__.__name__.lower()}")

    @abstractmethod
    def generate(self, result: dict[str, Any], output_path: Path):
        """Generate the output file.

        Args:
            result: Transcription result with speaker labels
            output_path: Path to save the output
        """
        pass


class TXTFormatter(BaseFormatter):
    """Plain text output formatter."""

    def generate(self, result: dict[str, Any], output_path: Path):
        """Generate a plain text transcript.

        Args:
            result: Transcription result with speaker labels
            output_path: Path to save the output
        """
        with output_path.open("w", encoding="utf-8") as txt_file:
            # First write the complete text
            txt_file.write(result["text"])
            txt_file.write("\n\n")

            # Then write with speaker labels
            txt_file.write("# Transcript with speakers\n\n")
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                txt_file.write(f"{speaker}: {text}\n")


class SRTFormatter(BaseFormatter):
    """SubRip (SRT) subtitle formatter."""

    def generate(self, result: dict[str, Any], output_path: Path):
        """Generate SRT subtitles.

        Args:
            result: Transcription result with speaker labels
            output_path: Path to save the output
        """
        with output_path.open("w", encoding="utf-8") as srt_file:
            for i, segment in enumerate(tqdm(result["segments"], desc="Writing SRT"), start=1):
                start = self._format_timestamp(float(segment["start"]))
                end = self._format_timestamp(float(segment["end"]))
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                srt_file.write(f"{i}\n{start} --> {end}\n{speaker}: {text}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for SRT format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string (HH:MM:SS,mmm)
        """
        milliseconds = int(seconds * 1000)
        hours = milliseconds // 3600000
        minutes = (milliseconds % 3600000) // 60000
        seconds = (milliseconds % 60000) // 1000
        ms = milliseconds % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"


class VTTFormatter(BaseFormatter):
    """WebVTT subtitle formatter."""

    def generate(self, result: dict[str, Any], output_path: Path):
        """Generate WebVTT subtitles.

        Args:
            result: Transcription result with speaker labels
            output_path: Path to save the output
        """
        with output_path.open("w", encoding="utf-8") as vtt_file:
            vtt_file.write("WEBVTT\n\n")
            for segment in tqdm(result["segments"], desc="Writing WebVTT"):
                start = self._format_timestamp(float(segment["start"]))
                end = self._format_timestamp(float(segment["end"]))
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                vtt_file.write(f"{start} --> {end}\n{speaker}: {text}\n\n")

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for WebVTT format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string (HH:MM:SS.mmm)
        """
        milliseconds = int(seconds * 1000)
        hours = milliseconds // 3600000
        minutes = (milliseconds % 3600000) // 60000
        seconds = (milliseconds % 60000) // 1000
        ms = milliseconds % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"


class TTMLFormatter(BaseFormatter):
    """TTML (IMSC1) subtitle formatter."""

    def generate(self, result: dict[str, Any], output_path: Path):
        """Generate TTML (IMSC1) subtitles.

        Args:
            result: Transcription result with speaker labels
            output_path: Path to save the output
        """
        ttml_title = self.config.ttml_title
        ttml_language = self.config.ttml_language

        with output_path.open("w", encoding="utf-8") as ttml_file:
            ttml_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            ttml_file.write(
                f'<tt xml:lang="{ttml_language}" xmlns="http://www.w3.org/ns/ttml" '
                f'xmlns:ttm="http://www.w3.org/ns/ttml#metadata" '
                f'xmlns:ttp="http://www.w3.org/ns/ttml#parameter" '
                f'xmlns:tts="http://www.w3.org/ns/ttml#styling" '
                f'xmlns:tt="http://www.w3.org/ns/ttml" xmlns:ims="http://www.w3.org/ns/ttml/profile/imsc1" '
                f'ttp:timeBase="media" ttp:frameRate="30" '
                f'ttp:profile="http://www.w3.org/ns/ttml/profile/imsc1/text">\n'
            )

            ttml_file.write("  <head>\n")
            ttml_file.write("    <metadata>\n")
            ttml_file.write(f"      <ttm:title>{xml_escape(ttml_title)}</ttm:title>\n")
            ttml_file.write("    </metadata>\n")
            ttml_file.write("    <styling>\n")
            ttml_file.write(
                '      <style xml:id="s1" tts:fontSize="100%" tts:color="white" tts:backgroundColor="black"/>\n'
            )
            ttml_file.write("    </styling>\n")
            ttml_file.write("    <layout>\n")
            ttml_file.write(
                '      <region xml:id="bottom" tts:origin="10% 80%" tts:extent="80% 20%" '
                'tts:displayAlign="after" tts:textAlign="center"/>\n'
            )
            ttml_file.write("    </layout>\n")
            ttml_file.write("  </head>\n")

            ttml_file.write("  <body>\n")
            ttml_file.write('    <div region="bottom" style="s1">\n')

            for segment in tqdm(result["segments"], desc="Writing TTML"):
                start = self._format_timestamp(float(segment["start"]))
                end = self._format_timestamp(float(segment["end"]))
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                # XML-escape the content
                full_text = xml_escape(f"{speaker}: {text}")
                ttml_file.write(f'      <p begin="{start}" end="{end}">{full_text}</p>\n')

            ttml_file.write("    </div>\n")
            ttml_file.write("  </body>\n")
            ttml_file.write("</tt>\n")

    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp for TTML format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted timestamp string (HH:MM:SS.mmm)
        """
        milliseconds = int(seconds * 1000)
        hours = milliseconds // 3600000
        minutes = (milliseconds % 3600000) // 60000
        seconds = (milliseconds % 60000) // 1000
        ms = milliseconds % 1000
        return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"
