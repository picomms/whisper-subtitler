"""
Tests for the CLI interface.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestCLI:
    """Test the real argparse CLI."""

    @patch("whisper_subtitler.modules.cli.setup_logging")
    @patch("whisper_subtitler.modules.cli.Application")
    @patch("whisper_subtitler.modules.cli.Config")
    def test_transcribe_passes_faster_whisper_args(
        self, mock_config_class, mock_app_class, mock_setup_logging, monkeypatch
    ):
        """Test real argparse plumbing for faster-whisper model/device/compute flags."""
        from whisper_subtitler.modules.cli import main

        mock_config = MagicMock()
        mock_config.model_size = "large-v3"
        mock_config.output_formats = ["json"]
        mock_config.skip_diarization = True
        mock_config.huggingface_token = None
        mock_config.load_from_env.return_value = mock_config
        mock_config.load_from_args.return_value = mock_config
        mock_config_class.return_value = mock_config

        mock_app = MagicMock()
        mock_app.process.return_value = {
            "results": {"input.mp3": {"json": "output.json"}},
            "failures": {},
        }
        mock_app_class.return_value = mock_app

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                "input.mp3",
                "-m",
                "large-v3",
                "--device",
                "cpu",
                "--compute-type",
                "int8",
                "--no-diarization",
            ],
        )

        assert main() == 0

        cli_args = mock_config.load_from_args.call_args[0][0]
        assert cli_args["model_size"] == "large-v3"
        assert cli_args["device"] == "cpu"
        assert cli_args["compute_type"] == "int8"
        assert cli_args["skip_diarization"] is True
        mock_setup_logging.assert_called_once_with(mock_config)
        mock_app.process.assert_called_once_with()

    @patch("whisper_subtitler.modules.cli.setup_logging")
    @patch("whisper_subtitler.modules.cli.Application")
    @patch("whisper_subtitler.modules.cli.Config")
    def test_transcribe_sets_json_format(self, mock_config_class, mock_app_class, mock_setup_logging, monkeypatch):
        """Test -f json overrides config.output_formats."""
        from whisper_subtitler.modules.cli import main

        mock_config = MagicMock()
        mock_config.model_size = "large-v3"
        mock_config.output_formats = ["txt"]
        mock_config.skip_diarization = True
        mock_config.huggingface_token = None
        mock_config.load_from_env.return_value = mock_config
        mock_config.load_from_args.return_value = mock_config
        mock_config_class.return_value = mock_config

        mock_app = MagicMock()
        mock_app.process.return_value = {
            "results": {"input.mp3": {"json": "output.json"}},
            "failures": {},
        }
        mock_app_class.return_value = mock_app

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                "input.mp3",
                "-f",
                "json",
                "--no-diarization",
            ],
        )

        assert main() == 0
        assert mock_config.output_formats == ["json"]

    @patch("whisper_subtitler.modules.cli.setup_logging")
    @patch("whisper_subtitler.modules.cli.Application")
    @patch("whisper_subtitler.modules.cli.Config")
    def test_transcribe_all_formats_includes_json(
        self, mock_config_class, mock_app_class, mock_setup_logging, monkeypatch
    ):
        """Test -f all expands to json plus subtitle formats."""
        from whisper_subtitler.modules.cli import ALL_OUTPUT_FORMATS, main

        mock_config = MagicMock()
        mock_config.model_size = "large-v3"
        mock_config.output_formats = ["json"]
        mock_config.skip_diarization = True
        mock_config.huggingface_token = None
        mock_config.load_from_env.return_value = mock_config
        mock_config.load_from_args.return_value = mock_config
        mock_config_class.return_value = mock_config

        mock_app = MagicMock()
        mock_app.process.return_value = {
            "results": {"input.mp3": {fmt: f"out.{fmt}" for fmt in ALL_OUTPUT_FORMATS}},
            "failures": {},
        }
        mock_app_class.return_value = mock_app

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                "input.mp3",
                "-f",
                "all",
                "--no-diarization",
            ],
        )

        assert main() == 0
        assert mock_config.output_formats == ALL_OUTPUT_FORMATS

    @patch("whisper_subtitler.modules.cli.setup_logging")
    @patch("whisper_subtitler.modules.cli.Application")
    @patch("whisper_subtitler.modules.cli.Config")
    def test_transcribe_directory_defaults_output_to_input_dir(
        self, mock_config_class, mock_app_class, mock_setup_logging, monkeypatch, tmp_path
    ):
        """Directory input defaults -o to the directory itself."""
        from whisper_subtitler.modules.cli import main

        media_dir = tmp_path / "meetings"
        media_dir.mkdir()

        mock_config = MagicMock()
        mock_config.model_size = "large-v3"
        mock_config.output_formats = ["json"]
        mock_config.skip_diarization = True
        mock_config.huggingface_token = None
        mock_config.load_from_env.return_value = mock_config
        mock_config.load_from_args.return_value = mock_config
        mock_config_class.return_value = mock_config

        mock_app = MagicMock()
        mock_app.process.return_value = {
            "results": {str(media_dir / "a.mp3"): {"json": "a.json"}},
            "failures": {},
        }
        mock_app_class.return_value = mock_app

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                str(media_dir),
                "--no-diarization",
            ],
        )

        assert main() == 0
        cli_args = mock_config.load_from_args.call_args[0][0]
        assert cli_args["output_dir"] == str(media_dir)

    @patch("whisper_subtitler.modules.cli.setup_logging")
    @patch("whisper_subtitler.modules.cli.Application")
    @patch("whisper_subtitler.modules.cli.Config")
    def test_transcribe_exits_nonzero_on_batch_failures(
        self, mock_config_class, mock_app_class, mock_setup_logging, monkeypatch
    ):
        """CLI exits 1 when process() reports any failures."""
        from whisper_subtitler.modules.cli import main

        mock_config = MagicMock()
        mock_config.model_size = "large-v3"
        mock_config.output_formats = ["json"]
        mock_config.skip_diarization = True
        mock_config.huggingface_token = None
        mock_config.load_from_env.return_value = mock_config
        mock_config.load_from_args.return_value = mock_config
        mock_config_class.return_value = mock_config

        mock_app = MagicMock()
        mock_app.process.return_value = {
            "results": {"ok.mp3": {"json": "ok.json"}},
            "failures": {"bad.mp3": "boom"},
        }
        mock_app_class.return_value = mock_app

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                "input.mp3",
                "--no-diarization",
            ],
        )

        assert main() == 1

    def test_removed_auto_model_flag_is_rejected(self, monkeypatch):
        """Removed feature-creep flags should fail argparse."""
        from whisper_subtitler.modules.cli import main

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "whisper-subtitler",
                "transcribe",
                "input.mp3",
                "--auto-model",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
