"""
Test configuration and shared fixtures for whisper-subtitler.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


@pytest.fixture
def sample_audio_file(temp_output_dir):
    """Create a temporary audio file for testing."""
    audio_path = temp_output_dir / "test_audio.wav"
    # Just create an empty file for testing
    audio_path.touch()
    yield audio_path


@pytest.fixture
def sample_video_file(temp_output_dir):
    """Create a temporary video file for testing."""
    video_path = temp_output_dir / "test_video.mp4"
    # Just create an empty file for testing
    video_path.touch()
    yield video_path


@pytest.fixture
def mock_env_variables(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "mock-token")
    monkeypatch.setenv("WHISPER_MODEL_SIZE", "tiny")
    monkeypatch.setenv("SHOW_SPEAKER_DEBUG", "False")
    monkeypatch.setenv("TTML_TITLE", "Test Transcription")
    monkeypatch.setenv("TTML_LANGUAGE", "en-US")
    monkeypatch.setenv("AUDIO_SAMPLE_RATE", "16000")
    monkeypatch.setenv("AUDIO_CHANNELS", "1")

    # Yield to test and then clean up if needed
    yield


@pytest.fixture
def mock_config():
    """Return a mock configuration for testing."""

    class MockConfig:
        def __init__(self):
            self.model_size = "tiny"
            self.language = "en"
            self.output_formats = ["json"]
            self.num_speakers = None
            self.huggingface_token = "mock-token"
            self.use_cuda = False
            self.device = "cpu"
            self.compute_type = None
            self.beam_size = 5
            self.best_of = 5
            self.temperature = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            self.initial_prompt = None
            self.condition_on_previous_text = False
            self.vad_filter = True
            self.compression_ratio_threshold = 2.4
            self.log_prob_threshold = -1.0
            self.no_speech_threshold = 0.6
            self.repetition_penalty = 1.0
            self.no_repeat_ngram_size = 0
            self.hallucination_silence_threshold = None
            self.vad_min_silence_duration_ms = None
            self.vad_speech_pad_ms = None
            self.verbose = False
            self.input_file = "mock_input.mp4"
            self.output_dir = "mock_output"
            self.force_overwrite = False
            self.log_level = "INFO"
            self.log_file = None
            self.skip_diarization = False
            self.min_speakers = None
            self.max_speakers = None
            self.ttml_title = "Transcription"
            self.ttml_language = "en-GB"
            self.audio_sample_rate = "16000"
            self.audio_channels = "1"

        def get(self, key, default=None):
            return getattr(self, key, default)

        def load_from_env(self, env_file=None):
            self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN", self.huggingface_token)
            self.model_size = os.environ.get("WHISPER_MODEL_SIZE", self.model_size)
            return self

        def load_from_args(self, args):
            for key, value in args.items():
                if hasattr(self, key):
                    setattr(self, key, value)
            return self

        def validate(self):
            if not self.huggingface_token:
                raise ValueError("HUGGINGFACE_TOKEN is required")
            return self

    return MockConfig()


@pytest.fixture
def mock_whisper_model():
    """Mock the Whisper model for testing."""
    model = MagicMock()
    model.transcribe.return_value = {
        "text": "This is a test transcription.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test",
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 4.0,
                "text": "transcription.",
            },
        ],
    }
    return model


@pytest.fixture
def mock_diarization_pipeline():
    """Mock the Pyannote diarization pipeline for testing."""
    pipeline = MagicMock()

    # Create a mock diarization result that will be returned when the pipeline is called
    diarization_result = MagicMock()

    # Setup the itertracks method to yield speaker segments
    diarization_result.itertracks.return_value = [
        (MagicMock(start=0.0, end=2.2), None, "SPEAKER_01"),
        (MagicMock(start=2.3, end=4.1), None, "SPEAKER_02"),
    ]

    # Make the pipeline return the diarization result when called
    pipeline.return_value = diarization_result

    return pipeline


@pytest.fixture
def sample_transcription_result():
    """Return a sample transcription result for testing."""
    return {
        "text": "This is a test transcription.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test",
                "speaker": None,  # Will be filled by diarization
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 4.0,
                "text": "transcription.",
                "speaker": None,  # Will be filled by diarization
            },
        ],
    }


@pytest.fixture
def sample_diarization_result():
    """Return a sample diarization result for testing."""
    return [{"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"}, {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"}]


@pytest.fixture
def sample_combined_result():
    """Return a sample result with transcription and diarization combined."""
    return {
        "text": "This is a test transcription.",
        "segments": [
            {
                "id": 0,
                "start": 0.0,
                "end": 2.0,
                "text": "This is a test",
                "speaker": "SPEAKER_01",
            },
            {
                "id": 1,
                "start": 2.5,
                "end": 4.0,
                "text": "transcription.",
                "speaker": "SPEAKER_02",
            },
        ],
    }


@pytest.fixture
def mock_subprocess(monkeypatch):
    """Mock subprocess for testing."""
    mock = MagicMock()
    mock.run.return_value = MagicMock(returncode=0)
    monkeypatch.setattr("subprocess.run", mock.run)
    return mock


@pytest.fixture
def mock_torch_cuda(monkeypatch):
    """Mock torch.cuda for testing."""
    cuda_mock = MagicMock()
    cuda_mock.is_available.return_value = True
    monkeypatch.setattr("torch.cuda", cuda_mock)
    return cuda_mock
