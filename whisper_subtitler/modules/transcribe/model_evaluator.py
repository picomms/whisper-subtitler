"""
Whisper model evaluation module.

This module provides functionality to evaluate and compare different
Whisper model configurations for transcription accuracy.
"""

import json
import time
from pathlib import Path
from typing import Any

import torch
import whisper
from jiwer import wer

from ..logger import get_logger


class ModelEvaluator:
    """Evaluator for Whisper model configurations.

    Evaluates and compares different Whisper model sizes and configurations
    to find the optimal balance between accuracy and performance.
    """

    def __init__(self, config):
        """Initialize the evaluator with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config
        self.device = "cuda" if config.use_cuda and torch.cuda.is_available() else "cpu"
        self.logger = get_logger("model_evaluator")
        self.models = {}
        self.results_dir = Path(config.output_directory or ".") / "model_evaluation"
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Available model sizes in ascending order of complexity/performance
        self.model_sizes = ["tiny", "base", "small", "medium", "large"]

    def load_model(self, model_size: str) -> Any:
        """Load a Whisper model of the specified size.

        Args:
            model_size: Size of the model to load (tiny, base, small, medium, large)

        Returns:
            Loaded Whisper model
        """
        if model_size not in self.models:
            self.logger.info(f"Loading Whisper model: {model_size}")
            self.models[model_size] = whisper.load_model(model_size, device=self.device)

            # Enable CUDA optimizations if available
            if self.device == "cuda":
                self.logger.debug("Enabling CUDA optimizations")
                torch.backends.cudnn.benchmark = True
                if hasattr(torch.backends, "cuda"):
                    if hasattr(torch.backends.cuda, "matmul"):
                        torch.backends.cuda.matmul.allow_tf32 = True
                if hasattr(torch.backends, "cudnn"):
                    torch.backends.cudnn.allow_tf32 = True

        return self.models[model_size]

    def evaluate_models(
        self, audio_path: str, reference_text: str | None = None, model_sizes: list[str] | None = None
    ) -> dict[str, Any]:
        """Evaluate different Whisper models on the given audio file.

        Args:
            audio_path: Path to the audio file to transcribe
            reference_text: Optional reference text for WER calculation
            model_sizes: Optional list of model sizes to evaluate (default: all sizes)

        Returns:
            Dictionary with evaluation results
        """
        audio_file = Path(audio_path)
        if not audio_file.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Determine which models to evaluate
        sizes_to_evaluate = model_sizes or self.model_sizes

        # Validate model sizes
        for size in sizes_to_evaluate:
            if size not in self.model_sizes:
                self.logger.warning(f"Unknown model size: {size}, skipping")
                sizes_to_evaluate.remove(size)

        self.logger.info(f"Evaluating models: {', '.join(sizes_to_evaluate)}")

        results = {
            "audio_file": str(audio_file),
            "models": {},
            "best_model": None,
            "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        best_wer = float("inf")
        best_model = None

        # Evaluate each model
        for model_size in sizes_to_evaluate:
            self.logger.info(f"Evaluating {model_size} model")

            model = self.load_model(model_size)

            # Measure performance
            start_time = time.time()
            try:
                transcription = model.transcribe(str(audio_path), language=self.config.language)
                duration = time.time() - start_time

                # Calculate WER if reference text is provided
                model_wer = None
                if reference_text:
                    model_wer = wer(reference_text, transcription["text"])

                    # Track best model
                    if model_wer < best_wer:
                        best_wer = model_wer
                        best_model = model_size

                # Store results
                results["models"][model_size] = {
                    "transcription": transcription["text"][:500] + "..."
                    if len(transcription["text"]) > 500
                    else transcription["text"],
                    "processing_time": duration,
                    "wer": model_wer,
                    "character_count": len(transcription["text"]),
                    "word_count": len(transcription["text"].split()),
                    "characters_per_second": len(transcription["text"]) / duration,
                    "words_per_second": len(transcription["text"].split()) / duration,
                }

                self.logger.info(
                    f"{model_size} processed in {duration:.2f}s | WER: {model_wer if model_wer else 'N/A'}"
                )

            except Exception as e:
                self.logger.error(f"Error evaluating {model_size} model: {e}")
                results["models"][model_size] = {"error": str(e)}

        # Record best model
        if best_model:
            results["best_model"] = {"size": best_model, "wer": best_wer}
            self.logger.info(f"Best model: {best_model} (WER: {best_wer:.4f})")

        # Save results to file
        self._save_results(results, audio_file.stem)

        return results

    def _save_results(self, results: dict[str, Any], file_stem: str) -> None:
        """Save evaluation results to a JSON file.

        Args:
            results: Evaluation results
            file_stem: Base name for the results file
        """
        output_path = self.results_dir / f"{file_stem}_evaluation.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        self.logger.info(f"Evaluation results saved to {output_path}")

    def find_optimal_model(self, audio_file: str, reference_text: str, criteria: str = "balanced") -> str:
        """Find the optimal model for the given audio based on specified criteria.

        Args:
            audio_file: Path to the audio file
            reference_text: Reference transcription text
            criteria: Optimization criteria: 'accuracy', 'speed', or 'balanced'

        Returns:
            Name of the optimal model size
        """
        results = self.evaluate_models(audio_file, reference_text)

        if not results["models"]:
            self.logger.warning("No models were successfully evaluated")
            return self.config.model_size  # Return default model size

        # Filter out models with errors
        valid_models = {k: v for k, v in results["models"].items() if "error" not in v}

        if not valid_models:
            self.logger.warning("All models encountered errors during evaluation")
            return self.config.model_size

        # Different optimization strategies
        if criteria == "accuracy":
            # Sort by WER (ascending)
            sorted_models = sorted(valid_models.items(), key=lambda x: x[1]["wer"] or float("inf"))
            best_model = sorted_models[0][0]
        elif criteria == "speed":
            # Sort by processing time (ascending)
            sorted_models = sorted(valid_models.items(), key=lambda x: x[1]["processing_time"])
            best_model = sorted_models[0][0]
        else:  # balanced
            # Create a score balancing WER and speed
            # Normalize values to 0-1 range
            wer_values = [m["wer"] or 1.0 for m in valid_models.values() if "wer" in m]
            time_values = [m["processing_time"] for m in valid_models.values()]

            if wer_values and time_values:
                max_wer = max(wer_values)
                max_time = max(time_values)

                scores = {}
                for name, model in valid_models.items():
                    # Lower is better for both metrics
                    wer_score = (model["wer"] or 1.0) / max_wer if max_wer > 0 else 0
                    time_score = model["processing_time"] / max_time if max_time > 0 else 0

                    # Combined score (0.7 * accuracy + 0.3 * speed)
                    scores[name] = (0.7 * wer_score) + (0.3 * time_score)

                # Get model with lowest score
                best_model = min(scores.items(), key=lambda x: x[1])[0]
            else:
                # Fallback to medium if metrics are missing
                best_model = "medium"

        self.logger.info(f"Optimal model for {criteria} criteria: {best_model}")
        return best_model
