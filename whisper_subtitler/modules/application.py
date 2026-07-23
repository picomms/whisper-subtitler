"""
Application orchestrator module.

This module provides the main application class that orchestrates
the entire transcription and diarization process.
"""

from pathlib import Path

from .audio import AudioExtractor
from .config import Config
from .diarisation import Diarizer
from .logger import setup_logging
from .output import OutputFormatter
from .transcribe import Transcriber


class Application:
    """Main application orchestrator.

    Coordinates all components of the whisper-subtitler
    pipeline including configuration, audio extraction,
    transcription, diarization, and output generation.
    """

    def __init__(self, config: Config | None = None):
        """Initialize the application with the given configuration.

        Args:
            config: Application configuration
        """
        self.config = config or Config()

        # Configure logging
        self.logger = setup_logging(self.config)

        # Initialize components to None (will be lazily loaded)
        self.audio_extractor = None
        self.transcriber = None
        self.diarizer = None
        self.output_formatter = None

    def initialize(self):
        """Initialize all components."""
        self.logger.info("Initializing application components")

        # Initialize components
        self.audio_extractor = AudioExtractor(self.config)
        self.transcriber = Transcriber(self.config)
        if not self.config.skip_diarization:
            self.diarizer = Diarizer(self.config)
        self.output_formatter = OutputFormatter(self.config)

        self.logger.debug("All components initialized")

    def process(self) -> dict[str, str]:
        """Process the input file with all components.

        Returns:
            Dictionary of output paths by format

        Raises:
            Exception: If any part of the process fails
        """
        try:
            # Initialize components if not already initialized
            if not self.audio_extractor or not self.transcriber or not self.output_formatter:
                self.initialize()

            # Extract audio from input file
            input_file = Path(self.config.input_file)
            if not input_file.exists():
                self.logger.error(f"Input file not found: {input_file}")
                raise FileNotFoundError(f"Input file not found: {input_file}")

            # Determine output directory
            output_dir = Path(self.config.output_dir) if self.config.output_dir else input_file.parent
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate output path without extension
            base_name = input_file.stem
            audio_file = output_dir / f"{base_name}.wav"

            # Extract audio if it doesn't exist or force overwrite is enabled
            if not audio_file.exists() or self.config.force_overwrite:
                self.logger.info(f"Extracting audio to: {audio_file}")
                self.audio_extractor.extract_audio(input_file=str(input_file), output_file=str(audio_file))
            else:
                self.logger.info(f"Using existing audio file: {audio_file}")

            # Run transcription
            self.logger.info("Starting transcription")
            self.logger.debug(f"Using model size: {self.config.model_size}")
            transcription = self.transcriber.transcribe(str(audio_file))

            # Run speaker diarization if not skipped
            if self.config.skip_diarization:
                self.logger.info("Skipping speaker diarization as requested")
                # Assign default speaker to all segments
                for segment in transcription["segments"]:
                    segment["speaker"] = "Speaker"
            else:
                # Initialize diarization pipeline
                self.logger.info("Initializing speaker diarization pipeline")
                self.diarizer.initialize_pipeline()

                # Run diarization
                self.logger.info(f"Running speaker diarization on {audio_file}")
                speaker_segments = self.diarizer.diarize(str(audio_file))

                # Combine results
                self.logger.info("Combining transcription and diarization results")
                self.logger.debug(f"Found {len(speaker_segments)} speaker segments")
                self.logger.debug(f"Transcription has {len(transcription['segments'])} segments")
                self.logger.info("Assigning speakers to transcription segments")
                combined_result = self.diarizer.assign_speakers_to_segments(transcription, speaker_segments)

            # Generate output files
            self.logger.info("Generating output files")
            output_files = self.output_formatter.generate_outputs(transcription, base_name)

            self.logger.info("Processing complete")
            return output_files

        except Exception as e:
            self.logger.error(f"Processing failed: {e!s}")
            raise
