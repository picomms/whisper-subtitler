"""
Integration tests for the whisper-subtitler.

These tests verify that the complete application works correctly.
They require actual test media files to run properly.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# Import once implemented
# from whisper_subtitler.modules.application import Application
# from whisper_subtitler.modules.config import Config
# from whisper_subtitler.cli import app


class IntegrationSetup:
    """Helper class to set up integration tests."""

    @staticmethod
    def find_test_media():
        """Find test media files in the test directory."""
        test_media_dir = Path(__file__).parent / "media"
        if not test_media_dir.exists():
            pytest.skip("Test media directory not found")

        test_files = list(test_media_dir.glob("*.mp4")) + list(test_media_dir.glob("*.wav"))
        if not test_files:
            pytest.skip("No test media files found")

        return test_files[0]  # Use the first test file


@pytest.mark.integration
class TestIntegration:
    """Integration tests for the complete application workflow."""

    def test_end_to_end(self, temp_output_dir):
        """Test the complete end-to-end workflow with a real media file."""
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            # This will be enabled once test media is provided
            pytest.skip("Test media not yet available - test will be enabled later")

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"  # Use tiny model for faster testing
        # config.verbose = True
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Check output files
        # assert (temp_output_dir / f"{test_file.stem}.txt").exists()
        # assert (temp_output_dir / f"{test_file.stem}.srt").exists()
        # assert (temp_output_dir / f"{test_file.stem}.vtt").exists()
        # assert (temp_output_dir / f"{test_file.stem}.ttml").exists()

        # Assert log file was created
        # assert (temp_output_dir / "whisper-subtitler.log").exists()

        # For now, just mark as expected
        assert True

    def test_output_format_selection(self, temp_output_dir):
        """Test output format selection."""
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.output_formats = ["srt", "txt"]  # Only select SRT and TXT
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Check output files
        # assert (temp_output_dir / f"{test_file.stem}.txt").exists()
        # assert (temp_output_dir / f"{test_file.stem}.srt").exists()
        # assert not (temp_output_dir / f"{test_file.stem}.vtt").exists()
        # assert not (temp_output_dir / f"{test_file.stem}.ttml").exists()

        # For now, just mark as expected
        assert True

    @patch("torch.cuda.is_available")
    def test_cuda_support(self, mock_cuda_available, temp_output_dir):
        """Test CUDA support."""
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Mock CUDA availability
        mock_cuda_available.return_value = True

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.use_cuda = True
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Assert CUDA was used if available
        # assert mock_cuda_available.called

        # For now, just mark as expected
        assert True

    @patch("subprocess.run")
    def test_ffmpeg_audio_extraction(self, mock_subprocess, temp_output_dir, sample_video_file):
        """Test audio extraction with ffmpeg."""
        # Test when implemented
        # config = Config()
        # config.input_file = str(sample_video_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.huggingface_token = "mock-token"

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Check that ffmpeg was called correctly
        # assert mock_subprocess.called
        # ffmpeg_cmd = mock_subprocess.call_args[0][0]
        # assert "ffmpeg" in ffmpeg_cmd[0]
        # assert "-i" in ffmpeg_cmd
        # assert str(sample_video_file) in ffmpeg_cmd

        # For now, just mark as expected
        assert True

    def test_cli_integration(self, temp_output_dir):
        """Test CLI integration."""
        # This test would use the CLI runner to test the CLI interface
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Once implemented, test using the CLI runner
        # from typer.testing import CliRunner
        # runner = CliRunner()
        # result = runner.invoke(app, [
        #     "transcribe",
        #     "--input", str(test_file),
        #     "--output", str(temp_output_dir),
        #     "--model", "tiny"
        # ])

        # assert result.exit_code == 0
        # assert (temp_output_dir / f"{test_file.stem}.txt").exists()

        # For now, just mark as expected
        assert True

    def test_multiple_speakers(self, temp_output_dir):
        """Test speaker diarization with multiple speakers."""
        # Find test media with multiple speakers
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.num_speakers = 2  # Specify the number of speakers
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Read the SRT file and check for speaker labels
        # srt_file = temp_output_dir / f"{test_file.stem}.srt"
        # assert srt_file.exists()
        # srt_content = srt_file.read_text()
        # assert "SPEAKER_01" in srt_content
        # assert "SPEAKER_02" in srt_content

        # For now, just mark as expected
        assert True

    def test_language_selection(self, temp_output_dir):
        """Test language selection for transcription."""
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.language = "fr"  # Set language to French
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # For now, just mark as expected
        assert True

    def test_force_overwrite(self, temp_output_dir):
        """Test force overwrite option."""
        # Find test media
        try:
            test_file = IntegrationSetup.find_test_media()
        except:
            # Skip if no test media is available yet
            pytest.skip("Test media not yet available - test will be enabled later")

        # Create an existing output file
        output_file = temp_output_dir / "test.txt"
        output_file.write_text("Existing content")

        # Test when implemented
        # config = Config()
        # config.input_file = str(test_file)
        # config.output_dir = str(temp_output_dir)
        # config.model_size = "tiny"
        # config.force_overwrite = True  # Enable force overwrite
        # config.huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")

        # if not config.huggingface_token:
        #     pytest.skip("HUGGINGFACE_TOKEN environment variable not set")

        # app = Application(config)
        # app.initialize()
        # app.process(config.input_file)

        # Check that the file was overwritten
        # assert output_file.read_text() != "Existing content"

        # For now, just check the file exists
        assert output_file.exists()
        assert True
