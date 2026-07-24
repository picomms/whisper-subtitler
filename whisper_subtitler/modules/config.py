"""
Configuration management for whisper-subtitler.

This module handles loading, validating, and providing access to
configuration settings from various sources including environment
variables, command line arguments, and configuration files.
"""

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Default temperature fallback sequence (enables compression-ratio / log-prob retries)
DEFAULT_TEMPERATURE: list[float] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]


def parse_temperature(value: str) -> float | list[float]:
    """Parse a temperature env/config value as a single float or comma-separated list."""
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise ValueError("empty temperature value")
    temps = [float(p) for p in parts]
    if len(temps) == 1:
        return temps[0]
    return temps


def _env_truthy(value: str | None, default: bool) -> bool:
    """Parse a boolean env value; fall back to default when unset."""
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes")


def _parse_optional_float(raw: str | None) -> float | None:
    """Parse an optional float; empty/none/null → None."""
    if raw is None or raw.strip() == "" or raw.strip().lower() in ("none", "null"):
        return None
    return float(raw)


class Config:
    """Configuration manager for whisper-subtitler.

    Handles loading configuration from multiple sources with the following precedence:
    1. Command line arguments
    2. Configuration file
    3. Environment variables
    4. Default values
    """

    def __init__(self):
        """Initialize configuration with default values."""
        # Transcription settings (faster-whisper)
        self.model_size: str = "large-v3"
        self.language: str | None = "en"  # Language code or None for auto-detection
        self.beam_size: int = 5  # Beam search width
        self.best_of: int = 5  # Number of samples for beam search
        # Temperature fallback list enables compression-ratio / log-prob retries
        self.temperature: float | list[float] = list(DEFAULT_TEMPERATURE)
        self.initial_prompt: str | None = None  # Initial prompt for transcription
        # Anti-hallucination / decoding knobs
        self.condition_on_previous_text: bool = False
        self.vad_filter: bool = True
        self.compression_ratio_threshold: float = 2.4
        self.log_prob_threshold: float = -1.0
        self.no_speech_threshold: float = 0.6
        self.repetition_penalty: float = 1.0
        self.no_repeat_ngram_size: int = 0
        self.hallucination_silence_threshold: float | None = None
        self.vad_min_silence_duration_ms: int | None = None
        self.vad_speech_pad_ms: int | None = None
        # Device: "auto" | "cpu" | "cuda" — CPU is first-class; CUDA optional
        self.device: str = "auto"
        self.compute_type: str | None = None  # None → int8 (cpu) / float16 (cuda)

        # Diarization settings (pyannote/speaker-diarization-3.1)
        self.num_speakers: int | None = None  # Exact count, or None for auto/bounds
        self.huggingface_token: str | None = None
        self.skip_diarization: bool = False  # Option to skip diarization completely
        self.min_speakers: int | None = None  # Minimum number of speakers to detect
        self.max_speakers: int | None = None  # Maximum number of speakers to detect

        # Performance settings
        self.use_cuda: bool = True

        # Output settings (JSON is the primary default)
        self.output_formats: list[str] = ["json"]
        self.ttml_title: str = "Transcription"
        self.ttml_language: str = "en-GB"

        # File settings
        self.input_file: str | None = None
        self.output_dir: str | None = None
        self.force_overwrite: bool = False

        # Audio extraction settings
        self.audio_sample_rate: str = "16000"
        self.audio_channels: str = "1"
        # New audio processing settings
        self.audio_conversion: bool = True  # Convert audio to optimal format
        self.audio_normalization: bool = True  # Normalize audio volume
        self.noise_reduction: bool = True  # Apply noise reduction
        self.high_pass_filter: bool = True  # Apply high-pass filter
        self.high_pass_cutoff: int = 80  # High-pass filter cutoff frequency (Hz)

        # Logging settings
        self.verbose: bool = False
        self.log_level: str = "INFO"
        self.log_file: str | None = None

    def load_from_env(self, env_file: str | None = None) -> "Config":
        """Load configuration from environment variables.

        Args:
            env_file: Optional path to .env file

        Returns:
            Self for method chaining
        """
        # Load environment variables from .env file if specified
        if env_file:
            if not Path(env_file).exists():
                logger.warning(f"Environment file not found: {env_file}")
            else:
                # Use override=True to ensure values from this file take precedence
                load_dotenv(env_file, override=True)
                logger.debug(f"Loaded configuration from {env_file}")
        else:
            # Load from default .env file
            load_dotenv()
            logger.debug("Loaded configuration from default .env file")

        # Transcription settings
        self.model_size = os.environ.get("WHISPER_MODEL_SIZE", self.model_size)
        if "WHISPER_LANGUAGE" in os.environ:
            self.language = os.environ.get("WHISPER_LANGUAGE")
            # Empty string or "auto" means auto-detection
            if self.language in ["", "auto", "null"]:
                self.language = None

        # New Whisper model configuration
        if "WHISPER_BEAM_SIZE" in os.environ:
            try:
                self.beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", self.beam_size))
            except ValueError:
                logger.warning("Invalid WHISPER_BEAM_SIZE value in environment, ignoring")

        if "WHISPER_BEST_OF" in os.environ:
            try:
                self.best_of = int(os.environ.get("WHISPER_BEST_OF", self.best_of))
            except ValueError:
                logger.warning("Invalid WHISPER_BEST_OF value in environment, ignoring")

        if "WHISPER_TEMPERATURE" in os.environ:
            try:
                self.temperature = parse_temperature(os.environ["WHISPER_TEMPERATURE"])
            except ValueError:
                logger.warning("Invalid WHISPER_TEMPERATURE value in environment, ignoring")

        self.initial_prompt = os.environ.get("WHISPER_INITIAL_PROMPT", self.initial_prompt)
        # Empty string means no prompt
        if self.initial_prompt == "":
            self.initial_prompt = None

        if "WHISPER_CONDITION_ON_PREVIOUS_TEXT" in os.environ:
            self.condition_on_previous_text = _env_truthy(
                os.environ.get("WHISPER_CONDITION_ON_PREVIOUS_TEXT"),
                self.condition_on_previous_text,
            )
        if "WHISPER_VAD_FILTER" in os.environ:
            self.vad_filter = _env_truthy(os.environ.get("WHISPER_VAD_FILTER"), self.vad_filter)

        if "WHISPER_COMPRESSION_RATIO_THRESHOLD" in os.environ:
            try:
                self.compression_ratio_threshold = float(os.environ["WHISPER_COMPRESSION_RATIO_THRESHOLD"])
            except ValueError:
                logger.warning("Invalid WHISPER_COMPRESSION_RATIO_THRESHOLD value in environment, ignoring")

        if "WHISPER_LOG_PROB_THRESHOLD" in os.environ:
            try:
                self.log_prob_threshold = float(os.environ["WHISPER_LOG_PROB_THRESHOLD"])
            except ValueError:
                logger.warning("Invalid WHISPER_LOG_PROB_THRESHOLD value in environment, ignoring")

        if "WHISPER_NO_SPEECH_THRESHOLD" in os.environ:
            try:
                self.no_speech_threshold = float(os.environ["WHISPER_NO_SPEECH_THRESHOLD"])
            except ValueError:
                logger.warning("Invalid WHISPER_NO_SPEECH_THRESHOLD value in environment, ignoring")

        if "WHISPER_REPETITION_PENALTY" in os.environ:
            try:
                self.repetition_penalty = float(os.environ["WHISPER_REPETITION_PENALTY"])
            except ValueError:
                logger.warning("Invalid WHISPER_REPETITION_PENALTY value in environment, ignoring")

        if "WHISPER_NO_REPEAT_NGRAM_SIZE" in os.environ:
            try:
                self.no_repeat_ngram_size = int(os.environ["WHISPER_NO_REPEAT_NGRAM_SIZE"])
            except ValueError:
                logger.warning("Invalid WHISPER_NO_REPEAT_NGRAM_SIZE value in environment, ignoring")

        if "WHISPER_HALLUCINATION_SILENCE_THRESHOLD" in os.environ:
            try:
                self.hallucination_silence_threshold = _parse_optional_float(
                    os.environ.get("WHISPER_HALLUCINATION_SILENCE_THRESHOLD")
                )
            except ValueError:
                logger.warning("Invalid WHISPER_HALLUCINATION_SILENCE_THRESHOLD value in environment, ignoring")

        if "WHISPER_VAD_MIN_SILENCE_DURATION_MS" in os.environ:
            raw = os.environ.get("WHISPER_VAD_MIN_SILENCE_DURATION_MS", "").strip()
            if raw == "" or raw.lower() in ("none", "null"):
                self.vad_min_silence_duration_ms = None
            else:
                try:
                    self.vad_min_silence_duration_ms = int(raw)
                except ValueError:
                    logger.warning("Invalid WHISPER_VAD_MIN_SILENCE_DURATION_MS value in environment, ignoring")

        if "WHISPER_VAD_SPEECH_PAD_MS" in os.environ:
            raw = os.environ.get("WHISPER_VAD_SPEECH_PAD_MS", "").strip()
            if raw == "" or raw.lower() in ("none", "null"):
                self.vad_speech_pad_ms = None
            else:
                try:
                    self.vad_speech_pad_ms = int(raw)
                except ValueError:
                    logger.warning("Invalid WHISPER_VAD_SPEECH_PAD_MS value in environment, ignoring")

        # Diarization settings
        self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", self.huggingface_token)
        if "NUM_SPEAKERS" in os.environ:
            try:
                num_speakers = os.environ.get("NUM_SPEAKERS")
                self.num_speakers = int(num_speakers) if num_speakers else None
            except ValueError:
                logger.warning("Invalid NUM_SPEAKERS value in environment, ignoring")

        self.skip_diarization = os.environ.get("SKIP_DIARIZATION", str(self.skip_diarization)).lower() in (
            "1",
            "true",
            "yes",
        )

        if "MIN_SPEAKERS" in os.environ:
            try:
                min_speakers = os.environ.get("MIN_SPEAKERS")
                self.min_speakers = int(min_speakers) if min_speakers else None
            except ValueError:
                logger.warning("Invalid MIN_SPEAKERS value in environment, ignoring")

        if "MAX_SPEAKERS" in os.environ:
            try:
                max_speakers = os.environ.get("MAX_SPEAKERS")
                self.max_speakers = int(max_speakers) if max_speakers else None
            except ValueError:
                logger.warning("Invalid MAX_SPEAKERS value in environment, ignoring")

        # Performance settings
        if "WHISPER_DEVICE" in os.environ or "DEVICE" in os.environ:
            self.device = (os.environ.get("WHISPER_DEVICE") or os.environ.get("DEVICE") or self.device).lower()
        if "WHISPER_COMPUTE_TYPE" in os.environ or "COMPUTE_TYPE" in os.environ:
            self.compute_type = os.environ.get("WHISPER_COMPUTE_TYPE") or os.environ.get("COMPUTE_TYPE")
        self.use_cuda = os.environ.get("USE_CUDA", str(self.use_cuda)).lower() in ("1", "true", "yes")
        # Back-compat: USE_CUDA=false forces CPU unless WHISPER_DEVICE already set
        if "USE_CUDA" in os.environ and not self.use_cuda and "WHISPER_DEVICE" not in os.environ:
            self.device = "cpu"

        # Output settings
        if "OUTPUT_FORMATS" in os.environ:
            self.output_formats = os.environ.get("OUTPUT_FORMATS", "").split(",")
        self.ttml_title = os.environ.get("TTML_TITLE", self.ttml_title)
        self.ttml_language = os.environ.get("TTML_LANGUAGE", self.ttml_language)

        # File settings
        self.input_file = os.environ.get("INPUT_FILE", self.input_file)
        self.output_dir = os.environ.get("OUTPUT_DIR", self.output_dir)
        self.force_overwrite = os.environ.get("FORCE_OVERWRITE", str(self.force_overwrite)).lower() in (
            "1",
            "true",
            "yes",
        )

        # Audio extraction settings
        self.audio_sample_rate = os.environ.get("AUDIO_SAMPLE_RATE", self.audio_sample_rate)
        self.audio_channels = os.environ.get("AUDIO_CHANNELS", self.audio_channels)

        # New audio processing settings
        self.audio_conversion = os.environ.get("AUDIO_CONVERSION", str(self.audio_conversion)).lower() in (
            "1",
            "true",
            "yes",
        )
        self.audio_normalization = os.environ.get("AUDIO_NORMALIZATION", str(self.audio_normalization)).lower() in (
            "1",
            "true",
            "yes",
        )
        self.noise_reduction = os.environ.get("NOISE_REDUCTION", str(self.noise_reduction)).lower() in (
            "1",
            "true",
            "yes",
        )
        self.high_pass_filter = os.environ.get("HIGH_PASS_FILTER", str(self.high_pass_filter)).lower() in (
            "1",
            "true",
            "yes",
        )
        if "HIGH_PASS_CUTOFF" in os.environ:
            try:
                self.high_pass_cutoff = int(os.environ.get("HIGH_PASS_CUTOFF", self.high_pass_cutoff))
            except ValueError:
                logger.warning("Invalid HIGH_PASS_CUTOFF value in environment, ignoring")

        # Logging settings
        self.verbose = os.environ.get("SHOW_SPEAKER_DEBUG", str(self.verbose)).lower() in ("1", "true", "yes")
        self.log_level = os.environ.get("LOG_LEVEL", self.log_level)
        self.log_file = os.environ.get("LOG_FILE", self.log_file)

        return self

    def load_from_file(self, config_file: str) -> "Config":
        """Load configuration from a configuration file.

        Args:
            config_file: Path to configuration file

        Returns:
            Self for method chaining

        Raises:
            FileNotFoundError: If the config file doesn't exist
        """
        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        # Simple config file format: key = value
        # Comments start with #
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert string values to appropriate types
                    if hasattr(self, key):
                        current_value = getattr(self, key)
                        if key == "temperature":
                            try:
                                setattr(self, key, parse_temperature(value))
                            except ValueError:
                                logger.warning(f"Invalid temperature value: {value}")
                        elif value.lower() in ["true", "yes", "1"]:
                            setattr(self, key, True)
                        elif value.lower() in ["false", "no", "0"]:
                            setattr(self, key, False)
                        elif value.lower() in ["none", "null", ""]:
                            setattr(self, key, None)
                        elif isinstance(current_value, int) or key == "num_speakers":
                            try:
                                setattr(self, key, int(value))
                            except ValueError:
                                logger.warning(f"Invalid integer value for {key}: {value}")
                        elif isinstance(current_value, float):
                            try:
                                setattr(self, key, float(value))
                            except ValueError:
                                logger.warning(f"Invalid float value for {key}: {value}")
                        elif isinstance(current_value, list):
                            setattr(self, key, value.split(","))
                        else:
                            setattr(self, key, value)

        return self

    def load_from_args(self, args: dict[str, Any]) -> "Config":
        """Load configuration from command-line arguments.

        Args:
            args: Dictionary of argument keys and values

        Returns:
            Self for method chaining
        """
        for key, value in args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)

        return self

    def validate(self) -> "Config":
        """Validate configuration values.

        Returns:
            Self for method chaining

        Raises:
            ValueError: If any required configuration is missing or invalid
        """
        # Validate required values
        if not self.input_file:
            raise ValueError("Input file is required")

        # Only require HuggingFace token if diarization is not skipped
        if not self.skip_diarization and not self.huggingface_token:
            logger.error(
                "HUGGINGFACE_TOKEN environment variable not set. Please set it with your token from https://hf.co/settings/tokens"
            )
            logger.error(
                "Also make sure to accept the user conditions at https://hf.co/pyannote/speaker-diarization-3.1"
            )
            logger.error("If you don't have a token, you can use --no-diarization to skip speaker diarization")
            raise ValueError("HUGGINGFACE_TOKEN is required for speaker diarization")

        if self.num_speakers is not None and (self.min_speakers is not None or self.max_speakers is not None):
            logger.warning("num_speakers is set; min_speakers/max_speakers will be ignored")

        if self.min_speakers is not None and self.max_speakers is not None and self.min_speakers > self.max_speakers:
            raise ValueError(
                f"min_speakers ({self.min_speakers}) cannot be greater than max_speakers ({self.max_speakers})"
            )

        # Validate model size (faster-whisper ids + "large" alias → large-v3)
        valid_model_sizes = [
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v1",
            "large-v2",
            "large-v3",
            "turbo",
            "distil-large-v3",
        ]
        if self.model_size not in valid_model_sizes:
            raise ValueError(f"Invalid model size: {self.model_size}. Must be one of {valid_model_sizes}")

        if self.device not in ("auto", "cpu", "cuda"):
            raise ValueError(f"Invalid device: {self.device}. Must be one of auto, cpu, cuda")

        valid_compute_types = ["int8", "int8_float16", "int16", "float16", "float32"]
        if self.compute_type is not None:
            self.compute_type = self.compute_type.lower()
            if self.compute_type not in valid_compute_types:
                raise ValueError(f"Invalid compute type: {self.compute_type}. Must be one of {valid_compute_types}")

        # Validate output formats
        valid_formats = ["json", "txt", "srt", "vtt", "ttml"]
        for fmt in self.output_formats:
            if fmt not in valid_formats:
                raise ValueError(f"Invalid output format: {fmt}. Must be one of {valid_formats}")

        # Create output directory if it doesn't exist
        if self.output_dir:
            os.makedirs(self.output_dir, exist_ok=True)

        return self

    def get(self, key: str, default: Any = None) -> Any:
        """Get a configuration value with a fallback default.

        Args:
            key: The configuration key to retrieve
            default: Default value if key doesn't exist

        Returns:
            The configuration value or default
        """
        return getattr(self, key, default)
