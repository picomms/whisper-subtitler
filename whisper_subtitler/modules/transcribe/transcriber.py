"""
Audio transcription module using Whisper.

This module handles the transcription of audio files using
OpenAI's Whisper model with configurable parameters.
"""

from pathlib import Path
from typing import Any

import torch
import whisper

from ..logger import get_logger
from .model_evaluator import ModelEvaluator


class Transcriber:
    """Audio transcription using Whisper models.

    Handles loading and using Whisper models for
    transcribing audio files with various configurations.
    """

    def __init__(self, config):
        """Initialize the transcriber with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.model_size = config.model_size
        self.language = config.language
        self.logger = get_logger("transcribe")
        self.auto_model_selection = config.auto_model_selection
        self.model_selection_criteria = config.model_selection_criteria
        self.optimize_for_audio = config.optimize_for_audio

        # Determine device for inference
        self.device = "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
        self.model = None
        self.model_evaluator = None

        # Whisper configuration
        self.transcription_options = {
            "language": self.language,
            "beam_size": config.beam_size,
            "best_of": config.best_of,
            "temperature": config.temperature,
            "initial_prompt": config.initial_prompt,
        }

        # Filter None values
        self.transcription_options = {k: v for k, v in self.transcription_options.items() if v is not None}

    def load_model(self) -> Any:
        """Load the Whisper model.

        Returns:
            Loaded Whisper model
        """
        if self.model is None:
            self.logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size, device=self.device)

            # Enable CUDA optimizations if available
            if self.device == "cuda":
                self.logger.debug("Enabling CUDA optimizations")
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends, "cuda"):
                    if hasattr(torch.backends.cuda, "matmul"):
                        torch.backends.cuda.matmul.allow_tf32 = True

                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True

        return self.model

    def _get_model_evaluator(self) -> ModelEvaluator:
        """Get or create the model evaluator.

        Returns:
            ModelEvaluator instance
        """
        if self.model_evaluator is None:
            self.model_evaluator = ModelEvaluator(self.config)
        return self.model_evaluator

    def find_optimal_model_size(self, audio_path: str, reference_text: str | None = None) -> str:
        """Find the optimal model size for the given audio.

        Args:
            audio_path: Path to the audio file
            reference_text: Optional reference text for WER calculation

        Returns:
            Optimal model size
        """
        evaluator = self._get_model_evaluator()

        # If reference text is provided, use it for accuracy comparison
        if reference_text:
            return evaluator.find_optimal_model(audio_path, reference_text, self.model_selection_criteria)

        # If no reference text, evaluate a sample of the audio with different models
        # and choose based on the specified criteria (without WER)
        results = evaluator.evaluate_models(audio_path, model_sizes=["tiny", "base", "medium"])

        # Default to medium if evaluation fails
        if not results["models"]:
            return "medium"

        # Choose based on criteria (without WER)
        valid_models = {k: v for k, v in results["models"].items() if "error" not in v}

        if self.model_selection_criteria == "speed":
            # Sort by processing time (ascending)
            sorted_models = sorted(valid_models.items(), key=lambda x: x[1]["processing_time"])
            return sorted_models[0][0]
        else:
            # For accuracy or balanced without reference, use larger model
            return "medium"

    def _optimize_for_audio_type(self, audio_path: str) -> dict[str, Any]:
        """Optimize transcription parameters based on audio characteristics.

        Args:
            audio_path: Path to the audio file

        Returns:
            Dictionary of optimized parameters
        """
        import librosa
        import numpy as np

        optimized_options = self.transcription_options.copy()

        try:
            # Load audio sample (first 30 seconds)
            y, sr = librosa.load(audio_path, sr=None, duration=30)

            # Check audio quality metrics
            rms = np.sqrt(np.mean(y**2))
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()

            # Detect low quality audio (noisy or low volume)
            if rms < 0.01 or spectral_bandwidth > 4000:
                self.logger.info("Detected noisy audio, adjusting parameters")
                # For noisy audio, use lower temperature for more predictable output
                optimized_options["temperature"] = 0.0
                # Increase beam size for better accuracy
                optimized_options["beam_size"] = 5

            # Detect speech-heavy audio
            non_silent = librosa.effects.split(y, top_db=30)
            speech_ratio = sum(end - start for start, end in non_silent) / len(y)

            if speech_ratio > 0.8:
                self.logger.info("Detected continuous speech, adjusting parameters")
                # For content-dense speech, increase best_of
                optimized_options["best_of"] = max(5, optimized_options.get("best_of", 5))

            return optimized_options

        except Exception as e:
            self.logger.warning(f"Error in audio optimization: {e}")
            return self.transcription_options

    def transcribe(self, audio_path: str, reference_text: str | None = None) -> dict[str, Any]:
        """Transcribe the given audio file.

        Args:
            audio_path: Path to the audio file
            reference_text: Optional reference text for model selection

        Returns:
            Dictionary containing transcription results
        """
        # Verify audio file exists
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Select optimal model if enabled
        if self.auto_model_selection:
            self.logger.info("Auto model selection enabled")
            self.model_size = self.find_optimal_model_size(audio_path, reference_text)
            self.logger.info(f"Selected model: {self.model_size}")

            # Clear existing model to load new one
            self.model = None

        # Ensure model is loaded
        model = self.load_model()

        self.logger.info(f"Transcribing: {audio_path}")

        # Get transcription options
        options = self.transcription_options.copy()

        # Optimize for audio type if enabled
        if self.optimize_for_audio:
            self.logger.info("Optimizing parameters for audio type")
            options = self._optimize_for_audio_type(audio_path)

        # Transcribe with configured options
        try:
            self.logger.debug(f"Transcription options: {options}")
            result = model.transcribe(str(audio_path), **options)

            # Initialize speaker field for each segment (will be filled by diarization)
            for segment in result["segments"]:
                segment["speaker"] = None

            return result
        except Exception as e:
            self.logger.error(f"Transcription error: {e!s}")
            raise
