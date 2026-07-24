"""
Speaker diarization module using Pyannote.audio.

Uses pyannote/speaker-diarization-3.1 with optional speaker count
bounds passed directly to the pipeline.
"""

from __future__ import annotations

import inspect
from typing import Any

import torch
from pyannote.audio import Pipeline

from ..logger import get_logger
from ..transcribe.transcriber import resolve_device

PIPELINE_MODEL = "pyannote/speaker-diarization-3.1"


def pipeline_auth_kwargs(token: str | None) -> dict[str, Any]:
    """Build auth kwargs compatible with pyannote 3.x (`use_auth_token`) and 4.x (`token`)."""
    try:
        params = inspect.signature(Pipeline.from_pretrained).parameters
    except (TypeError, ValueError):
        params = {}
    # Prefer the parameter the installed version actually accepts.
    if "use_auth_token" in params and "token" not in params:
        return {"use_auth_token": token}
    return {"token": token}


class Diarizer:
    """Speaker diarization using pyannote.audio 3.1."""

    def __init__(self, config: Any):
        """Initialize the diarizer with the given config.

        Args:
            config: Configuration object
        """
        self.config = config
        self.num_speakers = config.num_speakers
        self.min_speakers = config.min_speakers
        self.max_speakers = config.max_speakers
        self.huggingface_token = config.huggingface_token
        self.logger = get_logger("diarisation")

        requested_device = getattr(config, "device", None)
        self.device = resolve_device(config)
        if requested_device == "cuda" and self.device == "cpu":
            self.logger.warning("CUDA was requested but is not available; falling back to CPU")

        self.pipeline = None

    def initialize_pipeline(self) -> Any:
        """Initialize the diarization pipeline.

        Returns:
            Initialized pipeline
        """
        if self.pipeline is None:
            self.logger.info(f"Initializing speaker diarization pipeline: {PIPELINE_MODEL}")

            try:
                auth_kwargs = pipeline_auth_kwargs(self.huggingface_token)
                self.pipeline = Pipeline.from_pretrained(
                    PIPELINE_MODEL,
                    **auth_kwargs,
                )

                if self.device == "cuda":
                    self.logger.info("Using CUDA for diarization")
                    self.pipeline = self.pipeline.to(torch.device("cuda"))
                else:
                    self.logger.info("Using CPU for diarization")

            except Exception as e:
                self.logger.error(f"Failed to initialize diarization pipeline: {e!s}")
                raise

        return self.pipeline

    def _speaker_options(self) -> dict[str, int]:
        """Build pipeline kwargs for optional speaker constraints."""
        if self.num_speakers is not None:
            if self.min_speakers is not None or self.max_speakers is not None:
                self.logger.warning("num_speakers is set; ignoring min_speakers/max_speakers for pipeline call")
            return {"num_speakers": self.num_speakers}

        options: dict[str, int] = {}
        if self.min_speakers is not None:
            options["min_speakers"] = self.min_speakers
        if self.max_speakers is not None:
            options["max_speakers"] = self.max_speakers
        return options

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        """Diarize the given audio file to identify speakers.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of speaker segments with start, end, and speaker label
        """
        self.initialize_pipeline()

        self.logger.info(f"Running speaker diarization on {audio_path}")

        try:
            options = self._speaker_options()
            if options:
                self.logger.info(f"Speaker constraints: {options}")

            diarization = self.pipeline(audio_path, **options)

            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })

            unique_speakers = {s["speaker"] for s in speaker_segments}
            self.logger.info(f"Detected {len(unique_speakers)} speaker(s)")

            if self.config.verbose:
                for seg in speaker_segments[:10]:
                    self.logger.debug(f"{seg['speaker']}: {seg['start']:.2f} - {seg['end']:.2f}")
                if len(speaker_segments) > 10:
                    self.logger.debug(f"... and {len(speaker_segments) - 10} more segments")

            return speaker_segments

        except Exception as e:
            self.logger.error(f"Diarization error: {e!s}")
            raise

    def assign_speakers_to_segments(
        self, transcription_result: dict[str, Any], speaker_segments: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Assign speakers to transcription segments.

        Args:
            transcription_result: Whisper transcription result
            speaker_segments: List of speaker segments from diarization

        Returns:
            Updated transcription result with speaker labels
        """
        self.logger.info("Assigning speakers to transcription segments")

        # Make explicit copies so callers can rely on the returned result
        # as the only carrier of speaker assignment changes.
        result = transcription_result.copy()
        result["segments"] = [segment.copy() for segment in transcription_result["segments"]]

        for segment in result["segments"]:
            segment["speaker"] = self._find_best_speaker_for_segment(segment, speaker_segments)

        return result

    def _find_best_speaker_for_segment(
        self, whisper_segment: dict[str, Any], speaker_segments: list[dict[str, Any]]
    ) -> str:
        """Find the best speaker for a transcription segment.

        Args:
            whisper_segment: Transcription segment from Whisper
            speaker_segments: List of speaker segments from diarization

        Returns:
            Speaker label for the segment
        """
        # First try exact match at start point
        for seg in speaker_segments:
            if seg["start"] <= whisper_segment["start"] < seg["end"]:
                return seg["speaker"]

        # If no match, find segment with maximum overlap
        max_overlap = 0
        best_speaker = "Unknown"
        segment_start = whisper_segment["start"]
        segment_end = whisper_segment["end"]

        for seg in speaker_segments:
            overlap_start = max(seg["start"], segment_start)
            overlap_end = min(seg["end"], segment_end)
            overlap = max(0, overlap_end - overlap_start)

            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = seg["speaker"]

        return best_speaker
