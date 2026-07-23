"""
Tests for the transcription module (faster-whisper).
"""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from whisper_subtitler.modules.transcribe.transcriber import (
    Transcriber,
    resolve_compute_type,
    resolve_device,
    resolve_model_name,
)


class TestResolveHelpers:
    def test_resolve_model_name_large_alias(self):
        assert resolve_model_name("large") == "large-v3"
        assert resolve_model_name("large-v3") == "large-v3"
        assert resolve_model_name("medium") == "medium"

    def test_resolve_device_explicit_cpu(self, mock_config):
        mock_config.device = "cpu"
        assert resolve_device(mock_config) == "cpu"

    def test_resolve_compute_type_defaults(self, mock_config):
        mock_config.compute_type = None
        with patch("ctranslate2.get_supported_compute_types", side_effect=lambda d: {"int8", "float32"} if d == "cpu" else {"float16", "float32"}):
            assert resolve_compute_type(mock_config, "cpu") == "int8"
            assert resolve_compute_type(mock_config, "cuda") == "float16"
        mock_config.compute_type = "int8_float16"
        with patch("ctranslate2.get_supported_compute_types", return_value={"int8_float16", "float16"}):
            assert resolve_compute_type(mock_config, "cuda") == "int8_float16"

    def test_resolve_compute_type_falls_back_when_float16_unsupported(self, mock_config):
        """Pascal-class GPUs: prefer int8 over float32 for VRAM headroom."""
        mock_config.compute_type = None
        with patch(
            "ctranslate2.get_supported_compute_types",
            return_value={"float32", "int8_float32", "int8"},
        ):
            assert resolve_compute_type(mock_config, "cuda") == "int8"


class TestTranscriber:
    """Test the Transcriber class with mocked faster-whisper."""

    def test_initialization(self, mock_config):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None

        transcriber = Transcriber(mock_config)
        assert transcriber.config == mock_config
        assert transcriber.model_size == mock_config.model_size
        assert transcriber.language == mock_config.language
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"

    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_load_model(self, mock_whisper_model_cls, mock_config):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"

        mock_model = MagicMock()
        mock_whisper_model_cls.return_value = mock_model

        transcriber = Transcriber(mock_config)
        model = transcriber.load_model()
        assert model == mock_model
        mock_whisper_model_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_transcribe(self, mock_whisper_model_cls, mock_config, sample_audio_file):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"
        mock_config.language = "en"

        mock_model = MagicMock()
        segments = [
            SimpleNamespace(id=0, start=0.0, end=2.0, text=" This is a test"),
            SimpleNamespace(id=1, start=2.5, end=4.0, text=" transcription."),
        ]
        info = SimpleNamespace(language="en", language_probability=0.99)
        mock_model.transcribe.return_value = (iter(segments), info)
        mock_whisper_model_cls.return_value = mock_model

        transcriber = Transcriber(mock_config)
        result = transcriber.transcribe(str(sample_audio_file))

        assert result["text"] == "This is a test transcription."
        assert len(result["segments"]) == 2
        assert result["segments"][0]["start"] == 0.0
        assert result["segments"][0]["end"] == 2.0
        assert isinstance(result["segments"][0]["start"], float)
        assert result["segments"][0]["text"] == "This is a test"
        assert result["segments"][0]["speaker"] is None
        assert result["language"] == "en"
        mock_model.transcribe.assert_called_once()
        call_kwargs = mock_model.transcribe.call_args
        assert call_kwargs[0][0] == str(sample_audio_file)
        assert call_kwargs[1]["language"] == "en"
        assert "log_progress" in call_kwargs[1]

    @patch("whisper_subtitler.modules.transcribe.transcriber.sys.stderr")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_transcribe_log_progress_false_when_not_tty(
        self, mock_whisper_model_cls, mock_stderr, mock_config, sample_audio_file
    ):
        """Non-TTY stderr disables faster-whisper progress unless verbose."""
        mock_config.verbose = False
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"
        mock_config.language = "en"
        mock_stderr.isatty.return_value = False

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([]),
            SimpleNamespace(language="en", language_probability=0.9),
        )
        mock_whisper_model_cls.return_value = mock_model

        Transcriber(mock_config).transcribe(str(sample_audio_file))

        assert mock_model.transcribe.call_args[1]["log_progress"] is False

    @patch("whisper_subtitler.modules.transcribe.transcriber.sys.stderr")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_transcribe_log_progress_true_when_verbose(
        self, mock_whisper_model_cls, mock_stderr, mock_config, sample_audio_file
    ):
        """Verbose forces log_progress even without a TTY."""
        mock_config.verbose = True
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"
        mock_config.language = "en"
        mock_stderr.isatty.return_value = False

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([]),
            SimpleNamespace(language="en", language_probability=0.9),
        )
        mock_whisper_model_cls.return_value = mock_model

        Transcriber(mock_config).transcribe(str(sample_audio_file))

        assert mock_model.transcribe.call_args[1]["log_progress"] is True

    @patch("whisper_subtitler.modules.transcribe.transcriber.sys.stderr")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_transcribe_log_progress_true_on_tty(
        self, mock_whisper_model_cls, mock_stderr, mock_config, sample_audio_file
    ):
        """Interactive stderr enables log_progress."""
        mock_config.verbose = False
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"
        mock_config.language = "en"
        mock_stderr.isatty.return_value = True

        mock_model = MagicMock()
        mock_model.transcribe.return_value = (
            iter([]),
            SimpleNamespace(language="en", language_probability=0.9),
        )
        mock_whisper_model_cls.return_value = mock_model

        Transcriber(mock_config).transcribe(str(sample_audio_file))

        assert mock_model.transcribe.call_args[1]["log_progress"] is True

    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_transcribe_with_language_auto(self, mock_whisper_model_cls, mock_config, sample_audio_file):
        mock_config.language = None
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "tiny"

        mock_model = MagicMock()
        info = SimpleNamespace(language="en", language_probability=0.95)
        mock_model.transcribe.return_value = (iter([]), info)
        mock_whisper_model_cls.return_value = mock_model

        transcriber = Transcriber(mock_config)
        result = transcriber.transcribe(str(sample_audio_file))
        assert result["language"] == "en"
        assert "language" not in mock_model.transcribe.call_args[1]

    @patch("ctranslate2.get_supported_compute_types", return_value={"float16", "float32", "int8"})
    @patch("whisper_subtitler.modules.transcribe.transcriber.torch")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_cuda_support(self, mock_whisper_model_cls, mock_torch, mock_supported, mock_config):
        mock_config.use_cuda = True
        mock_config.device = "auto"
        mock_config.compute_type = None
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.model_size = "tiny"
        mock_torch.cuda.is_available.return_value = True
        mock_whisper_model_cls.return_value = MagicMock()

        transcriber = Transcriber(mock_config)
        assert transcriber.device == "cuda"
        assert transcriber.compute_type == "float16"
        transcriber.load_model()
        mock_whisper_model_cls.assert_called_once_with("tiny", device="cuda", compute_type="float16")

    @patch("whisper_subtitler.modules.transcribe.transcriber.torch")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_fallback_to_cpu(self, mock_whisper_model_cls, mock_torch, mock_config):
        mock_config.use_cuda = True
        mock_config.device = "auto"
        mock_config.compute_type = None
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.model_size = "tiny"
        mock_torch.cuda.is_available.return_value = False
        mock_whisper_model_cls.return_value = MagicMock()

        transcriber = Transcriber(mock_config)
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        transcriber.load_model()
        mock_whisper_model_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_large_alias_loads_large_v3(self, mock_whisper_model_cls, mock_config):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None
        mock_config.model_size = "large"
        mock_whisper_model_cls.return_value = MagicMock()

        Transcriber(mock_config).load_model()

        mock_whisper_model_cls.assert_called_once_with("large-v3", device="cpu", compute_type="int8")

    @patch("whisper_subtitler.modules.transcribe.transcriber.torch")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_explicit_cuda_falls_back_to_cpu_with_warning(
        self, mock_whisper_model_cls, mock_torch, mock_config, caplog
    ):
        mock_config.use_cuda = True
        mock_config.device = "cuda"
        mock_config.compute_type = None
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.model_size = "tiny"
        mock_torch.cuda.is_available.return_value = False
        mock_whisper_model_cls.return_value = MagicMock()

        with caplog.at_level(logging.WARNING):
            transcriber = Transcriber(mock_config)

        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert "CUDA was requested but is not available; falling back to CPU" in caplog.text
        transcriber.load_model()
        mock_whisper_model_cls.assert_called_once_with("tiny", device="cpu", compute_type="int8")

    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_explicit_compute_type_is_passed_through(self, mock_whisper_model_cls, mock_config):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = "float32"
        mock_config.model_size = "tiny"
        mock_whisper_model_cls.return_value = MagicMock()

        Transcriber(mock_config).load_model()

        mock_whisper_model_cls.assert_called_once_with("tiny", device="cpu", compute_type="float32")

    @patch("whisper_subtitler.modules.transcribe.transcriber.torch")
    @patch("whisper_subtitler.modules.transcribe.transcriber.WhisperModel")
    def test_cuda_oom_falls_back_to_cpu(self, mock_whisper_model_cls, mock_torch, mock_config, sample_audio_file):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cuda"
        mock_config.compute_type = "int8"
        mock_config.model_size = "tiny"
        mock_config.language = "en"
        mock_config.verbose = False
        mock_torch.cuda.is_available.return_value = True

        cuda_model = MagicMock()
        cuda_model.transcribe.side_effect = RuntimeError("CUDA failed with error out of memory")
        cpu_model = MagicMock()
        info = SimpleNamespace(language="en", language_probability=0.9)
        cpu_model.transcribe.return_value = (
            iter([SimpleNamespace(id=0, start=0.0, end=1.0, text=" hi")]),
            info,
        )
        mock_whisper_model_cls.side_effect = [cuda_model, cpu_model]

        with patch("ctranslate2.get_supported_compute_types", return_value={"int8", "float32"}):
            transcriber = Transcriber(mock_config)
            result = transcriber.transcribe(str(sample_audio_file))

        assert result["text"] == "hi"
        assert transcriber.device == "cpu"
        assert transcriber.compute_type == "int8"
        assert mock_whisper_model_cls.call_count == 2
        assert mock_whisper_model_cls.call_args_list[0].kwargs == {
            "device": "cuda",
            "compute_type": "int8",
        }
        assert mock_whisper_model_cls.call_args_list[1].kwargs == {
            "device": "cpu",
            "compute_type": "int8",
        }

    def test_error_handling_missing_file(self, mock_config):
        mock_config.beam_size = 5
        mock_config.best_of = 5
        mock_config.temperature = 0.0
        mock_config.initial_prompt = None
        mock_config.device = "cpu"
        mock_config.compute_type = None

        transcriber = Transcriber(mock_config)
        try:
            transcriber.transcribe("/nonexistent/audio.wav")
            raise AssertionError("Expected FileNotFoundError")
        except FileNotFoundError:
            pass
