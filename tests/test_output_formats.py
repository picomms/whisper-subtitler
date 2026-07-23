"""
Tests for the output formats module.
"""

import json

from whisper_subtitler.modules.output import (
    JSONFormatter,
    OutputFormatter,
    SRTFormatter,
    TTMLFormatter,
    TXTFormatter,
    VTTFormatter,
)


class TestOutputFormatter:
    """Test the OutputFormatter orchestration."""

    def test_initialization_defaults_to_json(self, mock_config):
        formatter = OutputFormatter(mock_config)
        assert formatter.config == mock_config
        assert formatter.output_formats == ["json"]
        assert set(formatter.formatters) == {"json"}

    def test_generate_json_only(self, mock_config, sample_combined_result, temp_output_dir):
        mock_config.output_dir = str(temp_output_dir)
        mock_config.output_formats = ["json"]
        mock_config.force_overwrite = True
        mock_config.input_file = "test_video.mp4"

        formatter = OutputFormatter(mock_config)
        outputs = formatter.generate_outputs(sample_combined_result, "test_video")

        assert set(outputs) == {"json"}
        assert outputs["json"].exists()
        assert not (temp_output_dir / "test_video.srt").exists()

    def test_selective_json_and_srt(self, mock_config, sample_combined_result, temp_output_dir):
        mock_config.output_dir = str(temp_output_dir)
        mock_config.output_formats = ["json", "srt"]
        mock_config.force_overwrite = True

        formatter = OutputFormatter(mock_config)
        outputs = formatter.generate_outputs(sample_combined_result, "test_video")

        assert set(outputs) == {"json", "srt"}
        assert outputs["json"].exists()
        assert outputs["srt"].exists()
        assert not (temp_output_dir / "test_video.vtt").exists()


class TestJSONFormatter:
    """Test JSONFormatter."""

    def test_generate_json(self, mock_config, sample_combined_result, temp_output_dir):
        output_path = temp_output_dir / "out.json"
        JSONFormatter(mock_config).generate(sample_combined_result, output_path)

        data = json.loads(output_path.read_text(encoding="utf-8"))
        assert data["text"] == "This is a test transcription."
        assert len(data["segments"]) == 2
        assert data["segments"][0]["start"] == 0.0
        assert data["segments"][0]["end"] == 2.0
        assert isinstance(data["segments"][0]["start"], float)
        assert data["segments"][0]["text"] == "This is a test"
        assert data["segments"][0]["speaker"] == "SPEAKER_01"
        assert data["segments"][1]["speaker"] == "SPEAKER_02"
        assert "id" not in data["segments"][0]


class TestTXTFormatter:
    """Test TXTFormatter."""

    def test_generate_txt(self, mock_config, sample_combined_result, temp_output_dir):
        output_path = temp_output_dir / "out.txt"
        TXTFormatter(mock_config).generate(sample_combined_result, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "This is a test transcription." in content
        assert "SPEAKER_01: This is a test" in content
        assert "SPEAKER_02: transcription." in content


class TestSRTFormatter:
    """Test SRTFormatter."""

    def test_generate_srt(self, mock_config, sample_combined_result, temp_output_dir):
        output_path = temp_output_dir / "out.srt"
        SRTFormatter(mock_config).generate(sample_combined_result, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert "1\n" in content
        assert "00:00:00,000 --> 00:00:02,000" in content
        assert "SPEAKER_01: This is a test" in content
        assert "SPEAKER_02: transcription." in content


class TestVTTFormatter:
    """Test VTTFormatter."""

    def test_generate_vtt(self, mock_config, sample_combined_result, temp_output_dir):
        output_path = temp_output_dir / "out.vtt"
        VTTFormatter(mock_config).generate(sample_combined_result, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert content.startswith("WEBVTT")
        assert "00:00:00.000 --> 00:00:02.000" in content
        assert "SPEAKER_01: This is a test" in content


class TestTTMLFormatter:
    """Test TTMLFormatter."""

    def test_generate_ttml(self, mock_config, sample_combined_result, temp_output_dir):
        output_path = temp_output_dir / "out.ttml"
        TTMLFormatter(mock_config).generate(sample_combined_result, output_path)

        content = output_path.read_text(encoding="utf-8")
        assert '<?xml version="1.0"' in content
        assert "SPEAKER_01: This is a test" in content
        assert 'begin="00:00:00.000"' in content
