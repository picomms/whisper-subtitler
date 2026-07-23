"""
Tests for audio extraction / media preparation.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from whisper_subtitler.modules.audio.extractor import AudioExtractor


class TestAudioExtractor:
    """Test AudioExtractor with mocked FFmpeg."""

    @patch("whisper_subtitler.modules.audio.extractor.subprocess.run")
    def test_extract_mp3_to_wav(self, mock_run, mock_config, temp_output_dir):
        mock_run.return_value = MagicMock(returncode=0)
        input_mp3 = temp_output_dir / "talk.mp3"
        input_mp3.touch()
        output_wav = temp_output_dir / "talk.wav"

        extractor = AudioExtractor(mock_config)
        result = extractor.extract_audio(str(input_mp3), str(output_wav))

        assert result == output_wav.resolve()
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "ffmpeg"
        assert "-i" in cmd
        assert str(input_mp3.resolve()) in cmd
        assert "-vn" in cmd
        assert "pcm_s16le" in cmd
        assert str(output_wav.resolve()) in cmd

    @patch("whisper_subtitler.modules.audio.extractor.subprocess.run")
    def test_same_path_wav_uses_temp_then_replace(self, mock_run, mock_config, temp_output_dir):
        mock_run.return_value = MagicMock(returncode=0)
        wav_path = temp_output_dir / "meeting.wav"
        wav_path.write_bytes(b"original")

        def fake_ffmpeg(cmd, check=True, capture_output=True, text=True):
            # FFmpeg writes the temp converted file
            out = Path(cmd[-1])
            out.write_bytes(b"converted")
            return MagicMock(returncode=0)

        mock_run.side_effect = fake_ffmpeg

        extractor = AudioExtractor(mock_config)
        result = extractor.extract_audio(str(wav_path), str(wav_path))

        assert result == wav_path.resolve()
        cmd = mock_run.call_args[0][0]
        ffmpeg_out = Path(cmd[-1])
        assert ffmpeg_out.name == "meeting.converted.wav"
        assert wav_path.read_bytes() == b"converted"
        assert not ffmpeg_out.exists()
