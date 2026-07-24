"""
Tests for the speaker diarization module (pyannote 3.1).
"""

import logging
from unittest.mock import MagicMock, patch

from whisper_subtitler.modules.diarisation.diarizer import PIPELINE_MODEL, Diarizer, pipeline_auth_kwargs


class TestPipelineAuthKwargs:
    def test_prefers_use_auth_token_when_only_that_exists(self):
        with patch(
            "whisper_subtitler.modules.diarisation.diarizer.inspect.signature",
            return_value=MagicMock(parameters={"checkpoint_path": None, "use_auth_token": None}),
        ):
            assert pipeline_auth_kwargs("hf_x") == {"use_auth_token": "hf_x"}

    def test_uses_token_when_available(self):
        with patch(
            "whisper_subtitler.modules.diarisation.diarizer.inspect.signature",
            return_value=MagicMock(parameters={"checkpoint": None, "token": None}),
        ):
            assert pipeline_auth_kwargs("hf_x") == {"token": "hf_x"}


class TestDiarizer:
    """Test the Diarizer class with a mocked pyannote pipeline."""

    def test_initialization(self, mock_config):
        diarizer = Diarizer(mock_config)
        assert diarizer.config == mock_config
        assert diarizer.num_speakers == mock_config.num_speakers
        assert diarizer.min_speakers == mock_config.min_speakers
        assert diarizer.max_speakers == mock_config.max_speakers
        assert diarizer.huggingface_token == mock_config.huggingface_token
        assert diarizer.device == "cpu"

    @patch("whisper_subtitler.modules.diarisation.diarizer.pipeline_auth_kwargs", return_value={"use_auth_token": "mock-token"})
    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_initialize_pipeline_uses_3_1_and_token(self, mock_from_pretrained, mock_auth_kwargs, mock_config):
        mock_pipeline = MagicMock()
        mock_from_pretrained.return_value = mock_pipeline

        diarizer = Diarizer(mock_config)
        pipeline = diarizer.initialize_pipeline()

        assert pipeline == mock_pipeline
        mock_auth_kwargs.assert_called_once_with(mock_config.huggingface_token)
        mock_from_pretrained.assert_called_once_with(
            PIPELINE_MODEL,
            use_auth_token=mock_config.huggingface_token,
        )
        mock_pipeline.to.assert_not_called()

    @patch("whisper_subtitler.modules.diarisation.diarizer.torch")
    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_initialize_pipeline_moves_to_cuda(self, mock_from_pretrained, mock_torch, mock_config):
        mock_config.device = "auto"
        mock_config.use_cuda = True
        mock_torch.cuda.is_available.return_value = True
        mock_torch.device.return_value = "cuda"
        # resolve_device imports torch from transcriber module
        with patch(
            "whisper_subtitler.modules.transcribe.transcriber.torch.cuda.is_available",
            return_value=True,
        ):
            mock_pipeline = MagicMock()
            mock_from_pretrained.return_value = mock_pipeline
            mock_pipeline.to.return_value = mock_pipeline

            diarizer = Diarizer(mock_config)
            assert diarizer.device == "cuda"
            diarizer.initialize_pipeline()

            mock_torch.device.assert_called_once_with("cuda")
            mock_pipeline.to.assert_called_once_with("cuda")

    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_diarize_extracts_segments(self, mock_from_pretrained, mock_config, sample_audio_file):
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.itertracks.return_value = [
            (MagicMock(start=0.0, end=2.2), None, "SPEAKER_01"),
            (MagicMock(start=2.3, end=4.1), None, "SPEAKER_02"),
        ]
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline

        diarizer = Diarizer(mock_config)
        segments = diarizer.diarize(str(sample_audio_file))

        assert segments == [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]
        mock_pipeline.assert_called_once_with(str(sample_audio_file))
        mock_result.itertracks.assert_called_once_with(yield_label=True)

    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_diarize_passes_num_speakers(self, mock_from_pretrained, mock_config, sample_audio_file):
        mock_config.num_speakers = 2
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.itertracks.return_value = []
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline

        Diarizer(mock_config).diarize(str(sample_audio_file))

        mock_pipeline.assert_called_once_with(str(sample_audio_file), num_speakers=2)

    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_diarize_passes_min_max_speakers(self, mock_from_pretrained, mock_config, sample_audio_file):
        mock_config.num_speakers = None
        mock_config.min_speakers = 2
        mock_config.max_speakers = 4
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.itertracks.return_value = []
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline

        Diarizer(mock_config).diarize(str(sample_audio_file))

        mock_pipeline.assert_called_once_with(
            str(sample_audio_file),
            min_speakers=2,
            max_speakers=4,
        )

    @patch("whisper_subtitler.modules.diarisation.diarizer.Pipeline.from_pretrained")
    def test_num_speakers_prefers_over_min_max(self, mock_from_pretrained, mock_config, sample_audio_file, caplog):
        mock_config.num_speakers = 3
        mock_config.min_speakers = 1
        mock_config.max_speakers = 5
        mock_pipeline = MagicMock()
        mock_result = MagicMock()
        mock_result.itertracks.return_value = []
        mock_pipeline.return_value = mock_result
        mock_from_pretrained.return_value = mock_pipeline

        with caplog.at_level(logging.WARNING):
            Diarizer(mock_config).diarize(str(sample_audio_file))

        mock_pipeline.assert_called_once_with(str(sample_audio_file), num_speakers=3)
        assert "ignoring min_speakers/max_speakers" in caplog.text

    @patch("whisper_subtitler.modules.transcribe.transcriber.torch")
    def test_explicit_cuda_falls_back_to_cpu_with_warning(self, mock_torch, mock_config, caplog):
        mock_config.device = "cuda"
        mock_torch.cuda.is_available.return_value = False

        with caplog.at_level(logging.WARNING):
            diarizer = Diarizer(mock_config)

        assert diarizer.device == "cpu"
        assert "CUDA was requested but is not available; falling back to CPU" in caplog.text

    def test_assign_speakers_to_segments(self, mock_config, sample_transcription_result, sample_diarization_result):
        diarizer = Diarizer(mock_config)
        result = diarizer.assign_speakers_to_segments(sample_transcription_result, sample_diarization_result)

        assert result["segments"][0]["speaker"] == "SPEAKER_01"
        assert result["segments"][1]["speaker"] == "SPEAKER_02"
        # Original input is not mutated
        assert sample_transcription_result["segments"][0]["speaker"] is None
