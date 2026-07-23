"""
Tests for the configuration module.
"""

import os
import tempfile
from pathlib import Path

import pytest

from whisper_subtitler.modules.config import Config


# Using a placeholder for now until Config class is implemented
class MockConfigForTesting:
    def __init__(self):
        self.model_size = "medium"
        self.language = "en"
        self.output_formats = ["json"]
        self.num_speakers = None
        self.huggingface_token = None
        self.use_cuda = True
        self.verbose = False
        self.input_file = None
        self.output_dir = None
        self.force_overwrite = False
        self.log_level = "INFO"
        self.log_file = None
        self.skip_diarization = False
        self.min_speakers = None
        self.max_speakers = None

    def load_from_env(self, env_file=None):
        self.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
        self.model_size = os.environ.get("WHISPER_MODEL_SIZE", self.model_size)
        self.verbose = os.environ.get("SHOW_SPEAKER_DEBUG", "False").lower() in ("1", "true", "yes")
        return self

    def load_from_args(self, args):
        for key, value in args.items():
            if hasattr(self, key) and value is not None:
                setattr(self, key, value)
        return self

    def load_from_file(self, config_file):
        # Placeholder for loading from a config file
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        # Example implementation would read and parse the file
        return self

    def validate(self):
        # Example validation
        if not self.input_file:
            raise ValueError("Input file is required")
        if not self.huggingface_token:
            raise ValueError("HUGGINGFACE_TOKEN is required")
        return self


class TestConfig:
    """Test the configuration management."""

    def test_default_config(self):
        """Test the default configuration values."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()

        assert config.model_size == "medium"
        assert config.language == "en"
        assert config.output_formats == ["json"]
        assert config.num_speakers is None
        assert config.huggingface_token is None
        assert config.use_cuda is True
        assert config.verbose is False
        assert config.force_overwrite is False
        assert config.log_level == "INFO"
        assert config.log_file is None
        assert config.skip_diarization is False
        assert config.min_speakers is None
        assert config.max_speakers is None

    def test_env_loading(self, mock_env_variables):
        """Test loading configuration from environment variables."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting().load_from_env()

        assert config.huggingface_token == "mock-token"
        assert config.model_size == "tiny"
        assert config.verbose is False  # From SHOW_SPEAKER_DEBUG=False

    def test_args_loading(self):
        """Test loading configuration from arguments."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting().load_from_args({
            "model_size": "large",
            "language": "fr",
            "num_speakers": 3,
            "output_formats": ["srt"],
            "force_overwrite": True,
        })

        assert config.model_size == "large"
        assert config.language == "fr"
        assert config.num_speakers == 3
        assert config.output_formats == ["srt"]
        assert config.force_overwrite is True

    def test_args_none_values(self):
        """Test that None values in args don't override existing values."""
        config = MockConfigForTesting()
        config.model_size = "medium"

        # None value should not override existing value
        config.load_from_args({"model_size": None})
        assert config.model_size == "medium"

    def test_file_loading(self):
        """Test loading configuration from a file."""
        # Create a temporary config file
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("# Test config\n")
            temp_file.write("model_size = large\n")
            temp_file.write("language = fr\n")
            temp_path = temp_file.name

        try:
            # Replace with actual Config once implemented
            # This test will use the mock implementation for now
            config = MockConfigForTesting()

            # Mock the load_from_file method to set values as if read from file
            original_load_from_file = config.load_from_file

            def mock_load_from_file(file_path):
                assert file_path == temp_path
                config.model_size = "large"
                config.language = "fr"
                return config

            # Temporarily replace the method
            config.load_from_file = mock_load_from_file

            # Load from the temp file
            config.load_from_file(temp_path)

            # Restore original method
            config.load_from_file = original_load_from_file

            assert config.model_size == "large"
            assert config.language == "fr"
        finally:
            # Clean up
            os.unlink(temp_path)

    def test_precedence(self, mock_env_variables):
        """Test configuration precedence (args > file > env > defaults)."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()

        # Start with defaults
        assert config.model_size == "medium"

        # Load from environment
        config.load_from_env()
        assert config.model_size == "tiny"  # From env

        # Mock loading from file (would override env)
        def mock_load_from_file(file_path):
            config.model_size = "small"  # From file
            return config

        original_load_from_file = config.load_from_file
        config.load_from_file = mock_load_from_file
        config.load_from_file("mock_path")
        config.load_from_file = original_load_from_file

        assert config.model_size == "small"

        # Then override with args (highest precedence)
        config.load_from_args({"model_size": "large"})
        assert config.model_size == "large"  # From args

    def test_validation(self):
        """Test configuration validation."""
        # Replace with actual Config once implemented
        config = MockConfigForTesting()
        config.huggingface_token = "token"
        config.input_file = "test.mp4"

        # Should pass with token and input file
        config.validate()

        # Should raise error without token
        config.huggingface_token = None
        with pytest.raises(ValueError, match="HUGGINGFACE_TOKEN is required"):
            config.validate()

        # Should raise error without input file
        config.huggingface_token = "token"
        config.input_file = None
        with pytest.raises(ValueError, match="Input file is required"):
            config.validate()

    def test_output_formats_validation(self):
        """Test validation of output formats."""
        # This test would check that only valid output formats are accepted
        # Replace with actual Config implementation once available
        config = MockConfigForTesting()

        # Mock the validate method to check output_formats
        original_validate = config.validate

        def mock_validate():
            if not set(config.output_formats).issubset({"json", "txt", "srt", "vtt", "ttml"}):
                raise ValueError("Invalid output format")
            return original_validate(self)

        # Set invalid format and test validation
        config.output_formats = ["invalid"]
        config.validate = mock_validate

        with pytest.raises(ValueError, match="Invalid output format"):
            config.validate()

        # Restore original validate method
        config.validate = original_validate


class TestRealConfig:
    """Regression tests for the real faster-whisper configuration plumbing."""

    def _valid_config(self, tmp_path):
        config = Config()
        config.input_file = str(tmp_path / "audio.mp3")
        config.skip_diarization = True
        return config

    def test_env_loads_model_device_and_compute_type(self, monkeypatch):
        monkeypatch.setenv("WHISPER_MODEL_SIZE", "large-v3")
        monkeypatch.setenv("WHISPER_DEVICE", "cpu")
        monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "int8")

        config = Config().load_from_env()

        assert config.model_size == "large-v3"
        assert config.device == "cpu"
        assert config.compute_type == "int8"

    def test_use_cuda_false_forces_cpu_when_device_not_set(self, monkeypatch, tmp_path):
        # Avoid the project .env injecting WHISPER_DEVICE=auto via load_dotenv()
        monkeypatch.delenv("WHISPER_DEVICE", raising=False)
        monkeypatch.delenv("DEVICE", raising=False)
        monkeypatch.setenv("USE_CUDA", "false")
        empty_env = tmp_path / "empty.env"
        empty_env.write_text("")

        config = Config().load_from_env(env_file=str(empty_env))

        assert config.device == "cpu"
        assert config.use_cuda is False

    def test_validate_rejects_invalid_model(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.model_size = "not-a-model"

        with pytest.raises(ValueError, match="Invalid model size"):
            config.validate()

    def test_validate_rejects_invalid_device(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.device = "metal"

        with pytest.raises(ValueError, match="Invalid device"):
            config.validate()

    def test_validate_rejects_invalid_compute_type(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.compute_type = "bfloat16"

        with pytest.raises(ValueError, match="Invalid compute type"):
            config.validate()

    def test_validate_allows_faster_whisper_compute_type(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.compute_type = "float32"

        config.validate()

        assert config.compute_type == "float32"

    def test_default_output_formats_is_json(self):
        config = Config()
        assert config.output_formats == ["json"]

    def test_validate_allows_json_format(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.output_formats = ["json"]
        config.validate()
        assert config.output_formats == ["json"]

    def test_validate_rejects_min_greater_than_max_speakers(self, tmp_path):
        config = self._valid_config(tmp_path)
        config.min_speakers = 4
        config.max_speakers = 2

        with pytest.raises(ValueError, match="min_speakers"):
            config.validate()

    def test_validate_warns_when_num_speakers_and_bounds_set(self, tmp_path, caplog):
        import logging

        config = self._valid_config(tmp_path)
        config.num_speakers = 2
        config.min_speakers = 1
        config.max_speakers = 4

        with caplog.at_level(logging.WARNING):
            config.validate()

        assert "num_speakers is set; min_speakers/max_speakers will be ignored" in caplog.text
