"""
Command-line interface for whisper-subtitler.

This module provides a command-line interface for the
whisper-subtitler application using argparse.
"""

import argparse
import logging
import sys
from pathlib import Path

from .application import Application
from .config import Config
from .logger import setup_logging

# Set up a module-level logger that will be initialized properly later
logger = logging.getLogger(__name__)


def main():
    """Entry point for the CLI."""
    global logger  # Use global logger in this function

    parser = argparse.ArgumentParser(description="Transcribe and diarize videos with OpenAI's Whisper and Pyannote")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe video file")
    transcribe_parser.add_argument("input_file", type=str, help="Input video file to transcribe")
    transcribe_parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        type=str,
        help="Output directory for generated files (defaults to input file directory)",
    )

    # Whisper model configuration
    transcribe_parser.add_argument(
        "-m",
        "--model",
        dest="model_size",
        choices=["tiny", "base", "small", "medium", "large"],
        default=None,
        help="Whisper model size to use (defaults to value in .env or 'medium')",
    )
    transcribe_parser.add_argument(
        "-l",
        "--language",
        dest="language",
        type=str,
        default=None,
        help="Language code for transcription (empty, 'auto', or None for auto-detection)",
    )
    transcribe_parser.add_argument(
        "--beam-size",
        dest="beam_size",
        type=int,
        help="Beam size for Whisper (default: 5)",
    )
    transcribe_parser.add_argument(
        "--best-of",
        dest="best_of",
        type=int,
        help="Number of candidates for Whisper (default: 5)",
    )
    transcribe_parser.add_argument(
        "--temperature",
        dest="temperature",
        type=float,
        help="Temperature for Whisper sampling (default: 0.0)",
    )
    transcribe_parser.add_argument(
        "--initial-prompt",
        dest="initial_prompt",
        type=str,
        help="Initial prompt to guide Whisper transcription",
    )
    transcribe_parser.add_argument(
        "--auto-model",
        dest="auto_model_selection",
        action="store_true",
        help="Automatically select optimal model based on audio",
    )
    transcribe_parser.add_argument(
        "--model-criteria",
        dest="model_selection_criteria",
        choices=["accuracy", "speed", "balanced"],
        help="Criteria for auto model selection (default: balanced)",
    )
    transcribe_parser.add_argument(
        "--optimize-audio",
        dest="optimize_for_audio",
        action="store_true",
        help="Optimize transcription parameters based on audio characteristics",
    )

    # Speaker configuration
    transcribe_parser.add_argument(
        "-s", "--speakers", dest="num_speakers", type=int, help="Number of speakers (if known)"
    )
    transcribe_parser.add_argument(
        "--min-speakers", dest="min_speakers", type=int, help="Minimum number of speakers to detect"
    )
    transcribe_parser.add_argument(
        "--max-speakers", dest="max_speakers", type=int, help="Maximum number of speakers to detect"
    )
    transcribe_parser.add_argument(
        "--similarity-threshold",
        dest="similarity_threshold",
        type=float,
        help="Threshold for speaker clustering (0.0-1.0, default: 0.85)",
    )

    # Output options
    transcribe_parser.add_argument(
        "-f",
        "--format",
        dest="formats",
        choices=["txt", "srt", "vtt", "ttml", "all"],
        nargs="+",
        default=None,
        help="Output formats to generate (defaults to all formats)",
    )

    # Audio processing options
    transcribe_parser.add_argument(
        "--normalize-audio",
        dest="audio_normalization",
        action="store_true",
        help="Normalize audio volume for better transcription",
    )
    transcribe_parser.add_argument(
        "--reduce-noise",
        dest="noise_reduction",
        action="store_true",
        help="Apply noise reduction to audio",
    )
    transcribe_parser.add_argument(
        "--high-pass",
        dest="high_pass_filter",
        action="store_true",
        help="Apply high-pass filter to reduce low frequency noise",
    )
    transcribe_parser.add_argument(
        "--cutoff",
        dest="high_pass_cutoff",
        type=int,
        help="High-pass filter cutoff frequency in Hz (default: 80)",
    )

    # General options
    transcribe_parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Enable verbose output")
    transcribe_parser.add_argument(
        "--cpu", dest="use_cuda", action="store_false", default=None, help="Force CPU only mode (no CUDA)"
    )
    transcribe_parser.add_argument(
        "--no-diarization",
        dest="skip_diarization",
        action="store_true",
        help="Skip speaker diarization (transcription only)",
    )
    transcribe_parser.add_argument(
        "--preprocess", dest="preprocess_audio", action="store_true", help="Preprocess audio for improved diarization"
    )
    transcribe_parser.add_argument(
        "--cluster",
        dest="cluster_speakers",
        action="store_true",
        help="Enable speaker clustering for improved consistency",
    )
    transcribe_parser.add_argument(
        "--optimize-speakers",
        dest="optimize_num_speakers",
        action="store_true",
        help="Optimize speaker count when number of speakers is known",
    )
    transcribe_parser.add_argument(
        "--voice-activity",
        dest="voice_activity_detection",
        action="store_true",
        help="Use voice activity detection for better speaker segments",
    )
    transcribe_parser.add_argument(
        "--min-silence",
        dest="min_silence_duration",
        type=float,
        help="Minimum silence duration in seconds (default: 0.3)",
    )
    transcribe_parser.add_argument(
        "--force", dest="force_overwrite", action="store_true", help="Force overwrite existing output files"
    )
    transcribe_parser.add_argument("--config", dest="config_file", type=str, help="Path to configuration file")
    transcribe_parser.add_argument(
        "--token",
        dest="huggingface_token",
        type=str,
        help="HuggingFace token for Pyannote (overrides environment variable)",
    )
    transcribe_parser.add_argument("--env-file", dest="env_file", type=str, help="Path to .env file")
    transcribe_parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Logging level",
    )
    transcribe_parser.add_argument("--log-file", dest="log_file", type=str, help="Path to log file")

    # Version command
    version_parser = subparsers.add_parser("version", help="Show version information")

    # Parse arguments
    args = parser.parse_args()

    # Handle version command
    if args.command == "version":
        try:
            from ..version import VERSION
        except ImportError:
            # Try direct import for when running directly
            from version import VERSION

        print(f"whisper-subtitler version {VERSION}")
        return 0

    # Handle transcribe command
    if args.command == "transcribe":
        try:
            # Build the configuration
            config = Config()

            # Enable early debug logging to see configuration loading
            if args.verbose:
                logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")
                logger = logging.getLogger(__name__)
                logger.debug("Early debug logging enabled")

            # Load from .env file if specified
            if args.env_file:
                logger.debug(f"Loading configuration from {args.env_file}")
                config.load_from_env(args.env_file)
            else:
                logger.debug("Loading configuration from default .env file")
                config.load_from_env()

            # Log the model size after loading from .env
            if args.verbose:
                logger.debug(f"Model size after loading .env: {config.model_size}")
                logger.debug(f"Skip diarization after loading .env: {config.skip_diarization}")

            # Load from config file if specified
            if args.config_file:
                config.load_from_file(args.config_file)

            # Process formats - only override if explicitly specified
            if args.formats:
                if "all" in args.formats:
                    output_formats = ["txt", "srt", "vtt", "ttml"]
                else:
                    output_formats = args.formats
                # Override the config
                config.output_formats = output_formats

            # Default output directory to same directory as input file if not specified
            input_path = Path(args.input_file)
            output_dir = args.output_dir
            if not output_dir:
                output_dir = str(input_path.parent)

            # Convert language = "" or "auto" to None for auto-detection
            language = args.language
            if language in ["", "auto", "null"]:
                language = None

            # Create a dictionary of only non-None CLI arguments to override config
            cli_args = {
                "input_file": args.input_file,
                "output_dir": output_dir,
            }

            # Only add other args if they're explicitly set by the user
            if args.model_size is not None:
                cli_args["model_size"] = args.model_size
            if language is not None:
                cli_args["language"] = language
            if args.num_speakers is not None:
                cli_args["num_speakers"] = args.num_speakers
            if args.min_speakers is not None:
                cli_args["min_speakers"] = args.min_speakers
            if args.max_speakers is not None:
                cli_args["max_speakers"] = args.max_speakers
            if args.similarity_threshold is not None:
                cli_args["similarity_threshold"] = args.similarity_threshold
            if args.beam_size is not None:
                cli_args["beam_size"] = args.beam_size
            if args.best_of is not None:
                cli_args["best_of"] = args.best_of
            if args.temperature is not None:
                cli_args["temperature"] = args.temperature
            if args.initial_prompt is not None:
                cli_args["initial_prompt"] = args.initial_prompt
            if args.high_pass_cutoff is not None:
                cli_args["high_pass_cutoff"] = args.high_pass_cutoff
            if args.min_silence_duration is not None:
                cli_args["min_silence_duration"] = args.min_silence_duration
            if args.model_selection_criteria is not None:
                cli_args["model_selection_criteria"] = args.model_selection_criteria

            # Boolean flags
            if args.verbose:
                cli_args["verbose"] = args.verbose
            if args.use_cuda is not None:
                cli_args["use_cuda"] = args.use_cuda
            if args.skip_diarization:
                cli_args["skip_diarization"] = args.skip_diarization
            if args.preprocess_audio:
                cli_args["preprocess_audio"] = args.preprocess_audio
            if args.cluster_speakers:
                cli_args["cluster_speakers"] = args.cluster_speakers
            if args.optimize_num_speakers:
                cli_args["optimize_num_speakers"] = args.optimize_num_speakers
            if args.force_overwrite:
                cli_args["force_overwrite"] = args.force_overwrite
            if args.auto_model_selection:
                cli_args["auto_model_selection"] = args.auto_model_selection
            if args.optimize_for_audio:
                cli_args["optimize_for_audio"] = args.optimize_for_audio
            if args.audio_normalization:
                cli_args["audio_normalization"] = args.audio_normalization
            if args.noise_reduction:
                cli_args["noise_reduction"] = args.noise_reduction
            if args.high_pass_filter:
                cli_args["high_pass_filter"] = args.high_pass_filter
            if args.voice_activity_detection:
                cli_args["voice_activity_detection"] = args.voice_activity_detection

            # Other fields
            if args.huggingface_token:
                cli_args["huggingface_token"] = args.huggingface_token
            if args.log_level:
                cli_args["log_level"] = args.log_level
            if args.log_file:
                cli_args["log_file"] = args.log_file

            # Override with command line arguments
            config.load_from_args(cli_args)

            # Set up logging
            log_handler = setup_logging(config)
            logger = logging.getLogger(__name__)

            logger.info(f"Processing {args.input_file}")
            logger.info(f"Using model: {config.model_size}")
            if config.skip_diarization:
                logger.info("Speaker diarization is disabled")
            elif config.huggingface_token:
                logger.info("Speaker diarization is enabled")

            # Log the model size to debug model loading
            logger.debug(f"Model size from CLI: {args.model_size}")
            logger.debug(f"Model size in config: {config.model_size}")

            # Create and run the application
            app = Application(config)
            output_files = app.process()

            # Print output file paths
            print("\nOutput files:")
            for fmt, path in output_files.items():
                print(f"  {fmt.upper()}: {path}")

            print("\nProcessing complete!")
            return 0

        except Exception as e:
            print(f"\nError: {e!s}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
