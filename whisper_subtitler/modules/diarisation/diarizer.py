"""
Speaker diarization module using Pyannote.audio.

This module handles the identification of different speakers
in audio files using the Pyannote.audio library.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from pyannote.audio import Pipeline
from sklearn.metrics.pairwise import cosine_similarity

from ..logger import get_logger


class Diarizer:
    """Speaker diarization using Pyannote.audio.

    Handles speaker identification in audio files with support for
    customized pipelines, speaker clustering, and known speaker count.
    """

    def __init__(self, config):
        """Initialize the diarizer with the given config.

        Args:
            config: Configuration object
        """
        self.config = config
        self.num_speakers = config.num_speakers
        self.min_speakers = config.min_speakers
        self.max_speakers = config.max_speakers
        self.similarity_threshold = config.similarity_threshold
        self.optimize_num_speakers = config.optimize_num_speakers
        self.cluster_speakers = config.cluster_speakers
        self.preprocess_audio = config.preprocess_audio
        self.huggingface_token = config.huggingface_token
        self.use_cuda = config.use_cuda and torch.cuda.is_available()

        # Initialize pipeline and target speakers attribute (used as fallback)
        self.pipeline = None
        self._target_speakers = None

        # Set up logging
        self.logger = get_logger("diarisation")

    def initialize_pipeline(self) -> Any:
        """Initialize the diarization pipeline.

        Returns:
            Initialized pipeline
        """
        if self.pipeline is None:
            self.logger.info("Initializing speaker diarization pipeline")

            try:
                # Load the pipeline from the pretrained model
                self.pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    use_auth_token=self.huggingface_token,
                )

                # Apply configuration if needed - Number of speakers
                if self.num_speakers is not None:
                    self.logger.info(f"Setting fixed number of speakers: {self.num_speakers}")
                    try:
                        # New pyannote.audio API
                        self.pipeline.instantiate({
                            "clustering": {
                                "num_clusters": self.num_speakers,
                            }
                        })
                    except Exception as e:
                        # For older versions or different API configurations
                        self.logger.warning(f"Failed to set num_clusters: {e}. Trying fallback method.")
                        # Store the num_speakers value and handle it during diarization
                        self._target_speakers = self.num_speakers

                # Move to GPU if available and configured
                if self.use_cuda and torch.cuda.is_available():
                    self.logger.info("Using CUDA for diarization")
                    cuda_device = torch.device("cuda")
                    self.pipeline = self.pipeline.to(cuda_device)

                    # Enable CUDA optimizations
                    torch.backends.cudnn.benchmark = True

            except Exception as e:
                self.logger.error(f"Failed to initialize diarization pipeline: {e!s}")
                raise

        return self.pipeline

    def preprocess_audio_file(self, audio_path: str) -> str:
        """Preprocess audio for improved diarization.

        Args:
            audio_path: Path to the audio file

        Returns:
            Path to the preprocessed audio file
        """
        import librosa
        import scipy.signal
        import soundfile as sf

        self.logger.info(f"Preprocessing audio: {audio_path}")

        # Generate output path
        input_path = Path(audio_path)
        output_path = input_path.with_suffix(".processed.wav")

        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)

            # Noise reduction
            y_reduced = librosa.effects.trim(y, top_db=20)[0]

            # Volume normalization
            y_normalized = librosa.util.normalize(y_reduced)

            # High-pass filter to remove low frequency noise
            b, a = scipy.signal.butter(4, 80 / (sr / 2), "highpass")
            y_filtered = scipy.signal.filtfilt(b, a, y_normalized)

            # Save processed audio
            sf.write(output_path, y_filtered, sr)
            self.logger.info(f"Preprocessed audio saved to {output_path}")

            return str(output_path)
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e!s}")
            # Fall back to original audio file
            return audio_path

    def diarize(self, audio_path: str) -> list[dict[str, Any]]:
        """Diarize the given audio file to identify speakers.

        Args:
            audio_path: Path to the audio file

        Returns:
            List of speaker segments with start, end, and speaker label
        """
        # Ensure pipeline is initialized
        self.initialize_pipeline()

        # Preprocess audio if enabled
        processed_path = audio_path
        if self.preprocess_audio:
            processed_path = self.preprocess_audio_file(audio_path)

        self.logger.info(f"Running speaker diarization on {processed_path}")

        try:
            # Run diarization
            diarization_options = {}

            # Apply fallback num_speakers if instantiate() failed but we have a target
            if self._target_speakers is not None and self.num_speakers is not None:
                self.logger.info(f"Using fallback method for {self._target_speakers} speakers")
                diarization_options["num_speakers"] = self._target_speakers

            # Run the pipeline with any provided options
            diarization = self.pipeline(processed_path, **diarization_options)

            # Extract speaker segments
            speaker_segments = []
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker,
                })

            # Apply speaker clustering if enabled
            if self.cluster_speakers:
                speaker_segments = self._apply_speaker_clustering(diarization, speaker_segments)

            # Apply fixed number of speakers if requested and not already handled
            if self.num_speakers is not None and self.optimize_num_speakers:
                speaker_segments = self._optimize_for_speaker_count(speaker_segments)

            unique_speakers = set(s["speaker"] for s in speaker_segments)
            self.logger.info(f"Detected {len(unique_speakers)} speaker(s)")

            if self.config.verbose:
                for seg in speaker_segments[:10]:  # Show only first 10 segments
                    self.logger.debug(f"{seg['speaker']}: {seg['start']:.2f} - {seg['end']:.2f}")
                if len(speaker_segments) > 10:
                    self.logger.debug(f"... and {len(speaker_segments) - 10} more segments")

            return speaker_segments

        except Exception as e:
            self.logger.error(f"Diarization error: {e!s}")
            raise

    def _apply_speaker_clustering(
        self, diarization_result, speaker_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply speaker clustering to merge similar speakers.

        Args:
            diarization_result: Raw diarization result
            speaker_segments: List of speaker segments

        Returns:
            Updated list of speaker segments with merged speakers
        """
        self.logger.info("Applying speaker clustering")

        try:
            # Extract voice embeddings for each speaker segment
            embeddings = {}  # Speaker ID to voice embedding
            has_valid_embeddings = False

            # In newer versions of pyannote.audio, we need to get embeddings differently
            try:
                # Try to get speaker data with the new API
                for track in diarization_result.itertracks():
                    # Unpack the tuple which might have different elements depending on API version
                    turn = track[0] if isinstance(track, tuple) else track
                    speaker = (
                        diarization_result.labels()[turn]
                        if callable(diarization_result.labels)
                        else diarization_result._labels[turn]
                    )

                    if speaker not in embeddings:
                        embeddings[speaker] = []

                    # Get the embedding for this segment using the pipeline's embedding model
                    try:
                        segment_audio = diarization_result.crop(turn)
                        if hasattr(self.pipeline, "embeddings") and hasattr(self.pipeline.embeddings, "crop"):
                            embedding = self.pipeline.embeddings.crop(segment_audio)
                            embeddings[speaker].append(embedding)
                            has_valid_embeddings = True
                        else:
                            self.logger.warning("Embeddings not available in pipeline")
                    except Exception as e:
                        self.logger.warning(f"Could not extract embedding for segment: {e}")

            except ValueError as e:
                self.logger.warning(f"Speaker clustering failed: {e}")
                return self._apply_alternative_clustering(diarization_result, speaker_segments)
            except Exception as e:
                self.logger.warning(f"Speaker clustering failed with unexpected error: {e}")
                return self._apply_alternative_clustering(diarization_result, speaker_segments)

            # If we couldn't extract any valid embeddings, use alternative clustering
            if not has_valid_embeddings:
                self.logger.warning("No valid embeddings could be extracted, using alternative clustering")
                return self._apply_alternative_clustering(diarization_result, speaker_segments)

            # Average embeddings per speaker
            for speaker in embeddings:
                if embeddings[speaker]:
                    embeddings[speaker] = np.mean(embeddings[speaker], axis=0)
                else:
                    self.logger.warning(f"No embeddings for speaker {speaker}, using alternative clustering")
                    return self._apply_alternative_clustering(diarization_result, speaker_segments)

            # Compute similarity matrix
            speakers = list(embeddings.keys())
            n_speakers = len(speakers)
            similarity_matrix = np.zeros((n_speakers, n_speakers))

            for i in range(n_speakers):
                for j in range(i + 1, n_speakers):
                    similarity = cosine_similarity(
                        embeddings[speakers[i]].reshape(1, -1), embeddings[speakers[j]].reshape(1, -1)
                    )[0][0]
                    similarity_matrix[i, j] = similarity
                    similarity_matrix[j, i] = similarity

            # Cluster speakers
            clusters = {}
            for i, speaker in enumerate(speakers):
                assigned = False
                for cluster_id, cluster_members in clusters.items():
                    similarities = [similarity_matrix[i, speakers.index(member)] for member in cluster_members]
                    if all(sim > self.similarity_threshold for sim in similarities):
                        clusters[cluster_id].append(speaker)
                        assigned = True
                        break

                if not assigned:
                    # Create new cluster
                    clusters[speaker] = [speaker]

            # Create mapping of original speakers to final speakers
            speaker_mapping = {}
            for cluster_id, members in clusters.items():
                for member in members:
                    speaker_mapping[member] = cluster_id

            # Apply mapping to speaker segments
            for segment in speaker_segments:
                segment["speaker"] = speaker_mapping.get(segment["speaker"], segment["speaker"])

            # Log results
            unique_speakers_after = set(s["speaker"] for s in speaker_segments)
            self.logger.info(f"Speaker clustering: {n_speakers} speakers -> {len(unique_speakers_after)} speakers")

            return speaker_segments

        except Exception as e:
            self.logger.warning(f"Speaker clustering failed: {e!s}")
            # Return original segments if clustering fails
            return speaker_segments

    def _apply_alternative_clustering(
        self, diarization_result, speaker_segments: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Apply an alternative clustering method based on temporal proximity.

        This is a fallback when embeddings are not available.

        Args:
            diarization_result: Raw diarization result
            speaker_segments: List of speaker segments

        Returns:
            Updated list of speaker segments with merged speakers
        """
        self.logger.info("Applying alternative speaker clustering")

        try:
            # Get speaker segments with their timestamps
            segments_by_speaker = {}

            # Initialize with default threshold if not set
            threshold = 0.85 if self.similarity_threshold is None else self.similarity_threshold

            # Handle cases where threshold might be invalid
            if not isinstance(threshold, (int, float)) or threshold <= 0 or threshold > 1.0:
                self.logger.warning(f"Invalid threshold value: {threshold}, using default 0.85")
                threshold = 0.85

            # Group segments by speaker
            for segment in speaker_segments:
                speaker = segment.get("speaker", "UNKNOWN")
                if speaker not in segments_by_speaker:
                    segments_by_speaker[speaker] = []
                segments_by_speaker[speaker].append(segment)

            # If number of speakers already matches target, no need to cluster
            current_speakers = len(segments_by_speaker)

            # Get target number of speakers, with fallbacks
            target_speakers = None
            if self._target_speakers is not None:
                target_speakers = self._target_speakers
            elif self.num_speakers is not None:
                target_speakers = self.num_speakers
            else:
                # Default to 2 speakers minimum
                target_speakers = max(2, current_speakers // 2)

            # Ensure target_speakers is an integer
            target_speakers = int(target_speakers)

            # No need to cluster if we already have the desired number of speakers
            if current_speakers <= target_speakers:
                self.logger.info(
                    f"Current speaker count {current_speakers} matches or is less than target {target_speakers}"
                )
                return speaker_segments

            # Simple clustering based on temporal proximity and speaking time
            # Calculate total speaking time for each speaker
            speaking_time = {}
            for speaker, segments in segments_by_speaker.items():
                speaking_time[speaker] = sum(s["end"] - s["start"] for s in segments)

            # Sort speakers by speaking time (descending)
            sorted_speakers = sorted(speaking_time.items(), key=lambda x: x[1], reverse=True)

            # Keep top N speakers based on target, merge the rest
            if len(sorted_speakers) > target_speakers:
                # Speakers to keep (with longest speaking time)
                speakers_to_keep = {s[0] for s in sorted_speakers[:target_speakers]}

                # Speakers to reassign (with shortest speaking time)
                speakers_to_reassign = [s[0] for s in sorted_speakers[target_speakers:]]

                # Create a mapping for speaker reassignment
                speaker_mapping = {}

                # For each speaker to reassign, find the closest main speaker
                for speaker_to_reassign in speakers_to_reassign:
                    if not segments_by_speaker.get(speaker_to_reassign):
                        continue

                    # Find closest main speaker by temporal proximity
                    closest_speaker = None
                    min_distance = float("inf")

                    # Get the average time of this speaker's segments
                    segments = segments_by_speaker[speaker_to_reassign]
                    avg_time = sum((s["start"] + s["end"]) / 2 for s in segments) / len(segments)

                    # Compare to each main speaker's average time
                    for main_speaker in speakers_to_keep:
                        if not segments_by_speaker.get(main_speaker):
                            continue

                        main_segments = segments_by_speaker[main_speaker]
                        main_avg_time = sum((s["start"] + s["end"]) / 2 for s in main_segments) / len(main_segments)

                        # Calculate temporal distance
                        distance = abs(avg_time - main_avg_time)

                        if distance < min_distance:
                            min_distance = distance
                            closest_speaker = main_speaker

                    # If found a main speaker, map to it
                    if closest_speaker:
                        speaker_mapping[speaker_to_reassign] = closest_speaker
                    else:
                        # If no main speaker found, keep the original one
                        speakers_to_keep.add(speaker_to_reassign)

                # Apply the mapping to create new segments
                new_segments = []
                for segment in speaker_segments:
                    new_segment = segment.copy()
                    speaker = segment["speaker"]

                    # Replace speaker if in the mapping
                    if speaker in speaker_mapping:
                        new_segment["speaker"] = speaker_mapping[speaker]

                    new_segments.append(new_segment)

                self.logger.info(
                    f"Alternative clustering reduced {current_speakers} speakers to {target_speakers} speakers"
                )
                return new_segments

            return speaker_segments

        except Exception as e:
            self.logger.warning(f"Alternative clustering failed: {e}. Returning original segments.")
            return speaker_segments

    def _optimize_for_speaker_count(self, speaker_segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Optimize speaker segments for a known number of speakers.

        Args:
            speaker_segments: List of speaker segments

        Returns:
            Updated list of speaker segments with optimized speakers
        """
        if self.num_speakers is None:
            return speaker_segments

        self.logger.info(f"Optimizing for {self.num_speakers} speakers")

        try:
            current_speakers = set(s["speaker"] for s in speaker_segments)
            num_current = len(current_speakers)

            if num_current == self.num_speakers:
                # Already have the correct number of speakers
                return speaker_segments

            elif num_current > self.num_speakers:
                # Need to merge speakers
                # Count speaking time for each speaker
                speaking_time = {}
                for segment in speaker_segments:
                    speaker = segment["speaker"]
                    duration = segment["end"] - segment["start"]
                    speaking_time[speaker] = speaking_time.get(speaker, 0) + duration

                # Sort speakers by speaking time (ascending)
                sorted_speakers = sorted(speaking_time.items(), key=lambda x: x[1])

                # Merge the speakers with least speaking time into the next speaker
                speakers_to_keep = set([s[0] for s in sorted_speakers[-self.num_speakers :]])
                speaker_mapping = {}

                # Speakers to merge (those not in speakers_to_keep)
                speakers_to_merge = [s[0] for s in sorted_speakers[: -self.num_speakers]]

                # Find nearest speakers in time for each speaker to merge
                for speaker_to_merge in speakers_to_merge:
                    # Find segments for this speaker
                    segments_for_speaker = [s for s in speaker_segments if s["speaker"] == speaker_to_merge]

                    # Find the closest speaker from speakers_to_keep
                    closest_speaker = None
                    min_distance = float("inf")

                    for segment in segments_for_speaker:
                        mid_point = (segment["start"] + segment["end"]) / 2

                        # Find nearest segment from a kept speaker
                        for other_segment in speaker_segments:
                            if other_segment["speaker"] in speakers_to_keep:
                                other_mid = (other_segment["start"] + other_segment["end"]) / 2
                                distance = abs(mid_point - other_mid)

                                if distance < min_distance:
                                    min_distance = distance
                                    closest_speaker = other_segment["speaker"]

                    # Map this speaker to the closest one
                    if closest_speaker:
                        speaker_mapping[speaker_to_merge] = closest_speaker
                    else:
                        # Fallback: map to the speaker with most talking time
                        speaker_mapping[speaker_to_merge] = sorted_speakers[-1][0]

                # Apply mapping
                for segment in speaker_segments:
                    if segment["speaker"] in speaker_mapping:
                        segment["speaker"] = speaker_mapping[segment["speaker"]]

            # Return the updated segments
            return speaker_segments

        except Exception as e:
            self.logger.warning(f"Speaker count optimization failed: {e!s}")
            # Return original segments if optimization fails
            return speaker_segments

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

        # Make a copy to avoid modifying the original
        result = transcription_result.copy()

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
