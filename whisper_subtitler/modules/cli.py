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

ALL_OUTPUT_FORMATS = ["json", "txt", "srt", "vtt", "ttml"]


def main():
    """Entry point for the CLI."""
    global logger  # Use global logger in this function

    parser = argparse.ArgumentParser(
        description="Transcribe and diarize audio or video with faster-whisper and Pyannote"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # Transcribe command
    transcribe_parser = subparsers.add_parser(
        "transcribe", help="Transcribe an audio/video file or a directory of media files"
    )
    transcribe_parser.add_argument(
        "input_file",
        type=str,
        help="Input audio/video file or directory of media files (e.g. mp3, wav, mp4)",
    )
    transcribe_parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        type=str,
        help="Output directory for generated files (defaults to the input file's directory, or the input directory itself)",
    )

    # Whisper model configuration
    transcribe_parser.add_argument(
        "-m",
        "--model",
        dest="model_size",
        choices=[
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
        ],
        default=None,
        help="Whisper model size (faster-whisper; default: large-v3 or .env)",
    )
    transcribe_parser.add_argument(
        "-l",
        "--language",
        dest="language",
        type=str,
        default=None,
        help="Language code for transcription (empty, 'auto', or None for auto-detection)",
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

    # Output options
    transcribe_parser.add_argument(
        "-f",
        "--format",
        dest="formats",
        choices=["json", "txt", "srt", "vtt", "ttml", "all"],
        nargs="+",
        default=None,
        help="Output formats to generate (default: json)",
    )

    # General options
    transcribe_parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", help="Enable verbose output")
    transcribe_parser.add_argument(
        "--device",
        dest="device",
        choices=["auto", "cpu", "cuda"],
        default=None,
        help="Inference device (default: auto — CUDA if available, else CPU)",
    )
    transcribe_parser.add_argument(
        "--cpu",
        dest="use_cuda",
        action="store_false",
        default=None,
        help="Force CPU only mode (alias for --device cpu)",
    )
    transcribe_parser.add_argument(
        "--compute-type",
        dest="compute_type",
        type=str,
        default=None,
        help="faster-whisper compute type (default: int8 on CPU, float16 on CUDA)",
    )
    transcribe_parser.add_argument(
        "--no-diarization",
        dest="skip_diarization",
        action="store_true",
        help="Skip speaker diarization (transcription only)",
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
    subparsers.add_parser("version", help="Show version information")

    # Parse arguments
    args = parser.parse_args()

    # Handle version command
    if args.command == "version":
        from ..version import VERSION

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
                    output_formats = list(ALL_OUTPUT_FORMATS)
                else:
                    output_formats = args.formats
                config.output_formats = output_formats

            # Default output directory: input dir itself if batching a directory,
            # otherwise the parent of the input file.
            input_path = Path(args.input_file)
            output_dir = args.output_dir
            if not output_dir:
                output_dir = str(input_path) if input_path.is_dir() else str(input_path.parent)

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

            # Boolean / optional flags
            if args.verbose:
                cli_args["verbose"] = args.verbose
            if getattr(args, "device", None) is not None:
                cli_args["device"] = args.device
            elif args.use_cuda is not None:
                # --cpu sets use_cuda False
                cli_args["use_cuda"] = args.use_cuda
                if not args.use_cuda:
                    cli_args["device"] = "cpu"
            if getattr(args, "compute_type", None) is not None:
                cli_args["compute_type"] = args.compute_type
            if args.skip_diarization:
                cli_args["skip_diarization"] = args.skip_diarization
            if args.force_overwrite:
                cli_args["force_overwrite"] = args.force_overwrite

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
            setup_logging(config)
            logger = logging.getLogger(__name__)

            logger.info(f"Processing {args.input_file}")
            logger.info(f"Using model: {config.model_size}")
            logger.info(f"Output formats: {', '.join(config.output_formats)}")
            if config.skip_diarization:
                logger.info("Speaker diarization is disabled")
            elif config.huggingface_token:
                logger.info("Speaker diarization is enabled")

            # Log the model size to debug model loading
            logger.debug(f"Model size from CLI: {args.model_size}")
            logger.debug(f"Model size in config: {config.model_size}")

            # Create and run the application (single file or sequential directory batch)
            app = Application(config)
            batch = app.process()
            results = batch.get("results", {})
            failures = batch.get("failures", {})

            if results:
                print("\nOutput files:")
                for input_path_str, output_files in results.items():
                    if len(results) > 1 or failures:
                        print(f"\n  {input_path_str}:")
                        indent = "    "
                    else:
                        indent = "  "
                    for fmt, path in output_files.items():
                        print(f"{indent}{fmt.upper()}: {path}")

            if failures:
                print("\nFailures:")
                for input_path_str, error in failures.items():
                    print(f"  {input_path_str}: {error}")
                print(f"\nCompleted with {len(failures)} failure(s), {len(results)} success(es).")
                return 1

            print("\nProcessing complete!")
            return 0

        except Exception as e:
            print(f"\nError: {e!s}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
