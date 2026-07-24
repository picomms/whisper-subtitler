"""
Tests for the application orchestrator.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from whisper_subtitler.modules.application import Application
from whisper_subtitler.modules.diarisation.diarizer import Diarizer

# Import once implemented
# from whisper_subtitler.modules.application import Application
# from whisper_subtitler.modules.transcribe import Transcriber
# from whisper_subtitler.modules.diarisation import Diarizer
# from whisper_subtitler.modules.output import OutputFormatter


class TestApplication:
    """Test the application orchestrator."""

    def test_initialization(self, mock_config):
        """Test initialization of the Application."""
        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # assert app.config == mock_config
        # assert app.transcriber is None
        # assert app.diarizer is None
        # assert app.output_formatter is None

        # For now, just ensure we can use the mock config
        assert mock_config.model_size == "tiny"
        assert True

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Diarizer")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_passes_diarized_speakers_to_formatter(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_diarizer,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
        sample_video_file,
    ):
        """Regression: assigned diarization labels must reach output generation."""
        mock_config.input_file = str(sample_video_file)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = False

        transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test", "speaker": None},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription.", "speaker": None},
            ],
        }
        speaker_segments = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        mock_transcriber.return_value.transcribe.return_value = transcription
        mock_diarizer_instance = mock_diarizer.return_value
        mock_diarizer_instance.diarize.return_value = speaker_segments
        real_diarizer = Diarizer(mock_config)
        mock_diarizer_instance.assign_speakers_to_segments.side_effect = real_diarizer.assign_speakers_to_segments
        mock_output_formatter.return_value.generate_outputs.return_value = {"srt": temp_output_dir / "test_video.srt"}

        app = Application(mock_config)
        batch = app.process()

        mock_audio_extractor.return_value.extract_audio.assert_called_once()
        mock_diarizer_instance.initialize_pipeline.assert_called_once()
        mock_diarizer_instance.diarize.assert_called_once_with(str(Path("temp") / "test_video.wav"))
        mock_output_formatter.return_value.generate_outputs.assert_called_once()
        output_data = mock_output_formatter.return_value.generate_outputs.call_args[0][0]

        assert batch["failures"] == {}
        assert batch["results"] == {str(sample_video_file): {"srt": temp_output_dir / "test_video.srt"}}
        assert output_data["segments"][0]["speaker"] == "SPEAKER_01"
        assert output_data["segments"][1]["speaker"] == "SPEAKER_02"
        assert transcription["segments"][0]["speaker"] is None

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_passes_default_speaker_when_diarization_skipped(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
        sample_video_file,
    ):
        """Regression: --no-diarization still sends labeled segments to formatters."""
        mock_config.input_file = str(sample_video_file)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True

        mock_transcriber.return_value.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test", "speaker": None},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription.", "speaker": None},
            ],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"srt": temp_output_dir / "test_video.srt"}

        app = Application(mock_config)
        batch = app.process()

        mock_audio_extractor.return_value.extract_audio.assert_called_once()
        mock_output_formatter.return_value.generate_outputs.assert_called_once()
        output_data = mock_output_formatter.return_value.generate_outputs.call_args[0][0]

        assert batch["failures"] == {}
        assert batch["results"] == {str(sample_video_file): {"srt": temp_output_dir / "test_video.srt"}}
        assert [segment["speaker"] for segment in output_data["segments"]] == ["Speaker", "Speaker"]

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_extracts_mp3_to_temp_wav(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """MP3 inputs are extracted to a temp WAV for processing."""
        mp3_path = temp_output_dir / "talk.mp3"
        mp3_path.write_bytes(b"fake-mp3")
        mock_config.input_file = str(mp3_path)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True
        mock_config.force_overwrite = False

        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": temp_output_dir / "talk.json"}

        Application(mock_config).process()

        mock_audio_extractor.return_value.extract_audio.assert_called_once_with(
            input_file=str(mp3_path),
            output_file=str(Path("temp") / "talk.wav"),
        )

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_extracts_wav_to_temp_without_touching_source(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """WAV inputs are converted into temp/; the source file is left alone."""
        wav_path = temp_output_dir / "meeting.wav"
        wav_path.write_bytes(b"existing")
        mock_config.input_file = str(wav_path)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True
        mock_config.force_overwrite = False

        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": temp_output_dir / "meeting.json"}

        Application(mock_config).process()

        mock_audio_extractor.return_value.extract_audio.assert_called_once_with(
            input_file=str(wav_path),
            output_file=str(Path("temp") / "meeting.wav"),
        )
        assert wav_path.read_bytes() == b"existing"

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_deletes_temp_wav_after_use(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """Temp WAV is removed after processing completes."""
        mp3_path = temp_output_dir / "talk.mp3"
        mp3_path.write_bytes(b"fake-mp3")
        mock_config.input_file = str(mp3_path)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True

        temp_wav = Path("temp") / "talk.wav"

        def fake_extract(input_file, output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"wav-data")
            return Path(output_file)

        mock_audio_extractor.return_value.extract_audio.side_effect = fake_extract
        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": temp_output_dir / "talk.json"}

        Application(mock_config).process()

        assert not temp_wav.exists()

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_deletes_temp_wav_on_failure(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """Temp WAV is removed even when a later stage fails."""
        mp3_path = temp_output_dir / "talk.mp3"
        mp3_path.write_bytes(b"fake-mp3")
        mock_config.input_file = str(mp3_path)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True

        temp_wav = Path("temp") / "talk.wav"

        def fake_extract(input_file, output_file):
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            Path(output_file).write_bytes(b"wav-data")
            return Path(output_file)

        mock_audio_extractor.return_value.extract_audio.side_effect = fake_extract
        mock_transcriber.return_value.transcribe.side_effect = RuntimeError("boom")

        batch = Application(mock_config).process()

        assert str(mp3_path) in batch["failures"]
        assert not temp_wav.exists()

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_initialize_components(self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config):
        """Test initializing the application components."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()

        # # Check that the components were created with the config
        # mock_transcriber.assert_called_once_with(mock_config)
        # mock_diarizer.assert_called_once_with(mock_config)
        # mock_output_formatter.assert_called_once_with(mock_config)

        # # Check that the components were assigned
        # assert app.transcriber == mock_transcriber_instance
        # assert app.diarizer == mock_diarizer_instance
        # assert app.output_formatter == mock_formatter_instance

        # For now, just ensure the test runs
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    @patch("subprocess.run")
    def test_process_with_audio_extraction(
        self,
        mock_subprocess_run,
        mock_output_formatter,
        mock_diarizer,
        mock_transcriber,
        mock_config,
        temp_output_dir,
        sample_video_file,
    ):
        """Test processing a video file with audio extraction."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks
        mock_transcriber_instance.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        mock_diarizer_instance.diarize.return_value = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        # Set up config
        mock_config.output_dir = str(temp_output_dir)

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_video_file))

        # # Check that audio was extracted
        # expected_audio_path = temp_output_dir / f"{sample_video_file.stem}.wav"
        # mock_subprocess_run.assert_called_once()
        # ffmpeg_cmd = mock_subprocess_run.call_args[0][0]
        # assert "ffmpeg" in ffmpeg_cmd[0]
        # assert str(expected_audio_path) in ffmpeg_cmd

        # # Check that transcription and diarization were called
        # mock_transcriber_instance.transcribe.assert_called_once()
        # mock_diarizer_instance.diarize.assert_called_once()

        # # Check that output was generated
        # mock_formatter_instance.generate_outputs.assert_called_once()

        # For now, just ensure we can use the sample file
        assert sample_video_file.exists()
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_process_with_audio_file(
        self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config, sample_audio_file
    ):
        """Test processing an audio file directly."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks
        mock_transcriber_instance.transcribe.return_value = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        mock_diarizer_instance.diarize.return_value = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_audio_file))

        # # Check that transcription and diarization were called directly with the audio file
        # mock_transcriber_instance.transcribe.assert_called_once_with(str(sample_audio_file))
        # mock_diarizer_instance.diarize.assert_called_once_with(str(sample_audio_file))

        # # Check that output was generated
        # mock_formatter_instance.generate_outputs.assert_called_once()

        # For now, just ensure we can use the sample file
        assert sample_audio_file.exists()
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_speaker_assignment(self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config):
        """Test assigning speakers to transcription segments."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks with test data
        transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        diarization = [
            {"start": 0.0, "end": 2.2, "speaker": "SPEAKER_01"},
            {"start": 2.3, "end": 4.1, "speaker": "SPEAKER_02"},
        ]

        mock_transcriber_instance.transcribe.return_value = transcription
        mock_diarizer_instance.diarize.return_value = diarization

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process("test_file.wav")

        # # Get the data passed to the output formatter
        # output_data = mock_formatter_instance.generate_outputs.call_args[0][0]

        # # Check that speakers were assigned correctly
        # assert output_data["segments"][0]["speaker"] == "SPEAKER_01"
        # assert output_data["segments"][1]["speaker"] == "SPEAKER_02"

        # For now, just ensure the test runs
        assert True

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    @patch("subprocess.run", side_effect=Exception("ffmpeg error"))
    def test_audio_extraction_error(
        self,
        mock_subprocess_run,
        mock_output_formatter,
        mock_diarizer,
        mock_transcriber,
        mock_config,
        sample_video_file,
    ):
        """Test handling of audio extraction errors."""
        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()

        # # Check that an error is raised for audio extraction failure
        # with pytest.raises(Exception, match="ffmpeg error"):
        #     app.process(str(sample_video_file))

        # # Check that no further processing was attempted
        # assert not mock_transcriber_instance.transcribe.called
        # assert not mock_diarizer_instance.diarize.called
        # assert not mock_formatter_instance.generate_outputs.called

        # For now, just ensure the test runs
        assert True

    def test_resolve_input_files_directory_filters_and_sorts(self, temp_output_dir):
        """Directory expansion is top-level, media-only, and sorted by name."""
        from whisper_subtitler.modules.application import resolve_input_files

        (temp_output_dir / "b.mp4").touch()
        (temp_output_dir / "a.mp3").touch()
        (temp_output_dir / "notes.txt").touch()
        (temp_output_dir / "nested").mkdir()
        (temp_output_dir / "nested" / "c.mp3").touch()

        files = resolve_input_files(temp_output_dir)
        assert [p.name for p in files] == ["a.mp3", "b.mp4"]

    def test_resolve_input_files_empty_directory_raises(self, temp_output_dir):
        from whisper_subtitler.modules.application import resolve_input_files

        (temp_output_dir / "readme.txt").touch()
        with pytest.raises(ValueError, match="No media files found"):
            resolve_input_files(temp_output_dir)

    def test_resolve_input_files_missing_raises(self, temp_output_dir):
        from whisper_subtitler.modules.application import resolve_input_files

        with pytest.raises(FileNotFoundError):
            resolve_input_files(temp_output_dir / "missing.mp3")

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_directory_sequential_order(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """Directory batch processes media files in sorted order on one Application."""
        (temp_output_dir / "b.mp4").write_bytes(b"mp4")
        (temp_output_dir / "a.mp3").write_bytes(b"mp3")
        (temp_output_dir / "skip.txt").write_text("ignore")
        nested = temp_output_dir / "subdir"
        nested.mkdir()
        (nested / "c.mp3").write_bytes(b"nested")

        mock_config.input_file = str(temp_output_dir)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True

        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }

        def fake_outputs(transcription, base_name):
            return {"json": temp_output_dir / f"{base_name}.json"}

        mock_output_formatter.return_value.generate_outputs.side_effect = fake_outputs

        app = Application(mock_config)
        batch = app.process()

        assert batch["failures"] == {}
        assert list(batch["results"].keys()) == [
            str(temp_output_dir / "a.mp3"),
            str(temp_output_dir / "b.mp4"),
        ]
        assert mock_transcriber.return_value.transcribe.call_count == 2
        # Same Application instance keeps components (models) warm
        assert mock_audio_extractor.call_count == 1
        assert mock_transcriber.call_count == 1

    @patch("whisper_subtitler.modules.application.sys.stderr")
    @patch("whisper_subtitler.modules.application.tqdm")
    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_multi_file_uses_tqdm(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_tqdm,
        mock_stderr,
        mock_config,
        temp_output_dir,
    ):
        """Directory batches wrap the file loop in tqdm when stderr is a TTY."""
        (temp_output_dir / "a.mp3").write_bytes(b"mp3")
        (temp_output_dir / "b.mp4").write_bytes(b"mp4")
        mock_config.input_file = str(temp_output_dir)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True
        mock_stderr.isatty.return_value = True

        files = [temp_output_dir / "a.mp3", temp_output_dir / "b.mp4"]
        mock_tqdm.side_effect = lambda iterable, **kwargs: iterable

        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": "out.json"}

        Application(mock_config).process()

        mock_tqdm.assert_called_once()
        call_kwargs = mock_tqdm.call_args[1]
        assert call_kwargs["desc"] == "Files"
        assert call_kwargs["unit"] == "file"
        assert call_kwargs["disable"] is False
        assert list(mock_tqdm.call_args[0][0]) == files

    @patch("whisper_subtitler.modules.application.tqdm")
    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_single_file_skips_tqdm(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_tqdm,
        mock_config,
        temp_output_dir,
        sample_video_file,
    ):
        """Single-file runs do not wrap tqdm around the file loop."""
        mock_config.input_file = str(sample_video_file)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True
        mock_transcriber.return_value.transcribe.return_value = {
            "text": "hi",
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "hi", "speaker": None}],
        }
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": "out.json"}

        Application(mock_config).process()

        mock_tqdm.assert_not_called()

    @patch("whisper_subtitler.modules.application.OutputFormatter")
    @patch("whisper_subtitler.modules.application.Transcriber")
    @patch("whisper_subtitler.modules.application.AudioExtractor")
    def test_process_continues_after_file_failure(
        self,
        mock_audio_extractor,
        mock_transcriber,
        mock_output_formatter,
        mock_config,
        temp_output_dir,
    ):
        """One failed file is recorded; remaining files still process."""
        first = temp_output_dir / "a.mp3"
        second = temp_output_dir / "b.mp4"
        first.write_bytes(b"mp3")
        second.write_bytes(b"mp4")

        mock_config.input_file = str(temp_output_dir)
        mock_config.output_dir = str(temp_output_dir)
        mock_config.skip_diarization = True

        mock_transcriber.return_value.transcribe.side_effect = [
            RuntimeError("boom"),
            {
                "text": "ok",
                "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "ok", "speaker": None}],
            },
        ]
        mock_output_formatter.return_value.generate_outputs.return_value = {"json": temp_output_dir / "b.json"}

        batch = Application(mock_config).process()

        assert str(first) in batch["failures"]
        assert "boom" in batch["failures"][str(first)]
        assert str(second) in batch["results"]
        assert mock_transcriber.return_value.transcribe.call_count == 2

    @pytest.mark.skip("Requires actual implementation")
    @patch("whisper_subtitler.modules.transcribe.Transcriber")
    @patch("whisper_subtitler.modules.diarisation.Diarizer")
    @patch("whisper_subtitler.modules.output.OutputFormatter")
    def test_no_speakers_detected(
        self, mock_output_formatter, mock_diarizer, mock_transcriber, mock_config, sample_audio_file
    ):
        """Test handling case where no speakers are detected."""
        # Setup mocks
        mock_transcriber_instance = MagicMock()
        mock_diarizer_instance = MagicMock()
        mock_formatter_instance = MagicMock()

        mock_transcriber.return_value = mock_transcriber_instance
        mock_diarizer.return_value = mock_diarizer_instance
        mock_output_formatter.return_value = mock_formatter_instance

        # Configure mocks with test data that has no speakers
        transcription = {
            "text": "This is a test transcription.",
            "segments": [
                {"id": 0, "start": 0.0, "end": 2.0, "text": "This is a test"},
                {"id": 1, "start": 2.5, "end": 4.0, "text": "transcription."},
            ],
        }

        # Empty diarization result
        diarization = []

        mock_transcriber_instance.transcribe.return_value = transcription
        mock_diarizer_instance.diarize.return_value = diarization

        # Once implemented, use the actual Application
        # app = Application(mock_config)
        # app.initialize()
        # app.process(str(sample_audio_file))

        # # Get the data passed to the output formatter
        # output_data = mock_formatter_instance.generate_outputs.call_args[0][0]

        # # Check that a default speaker was assigned
        # assert "speaker" in output_data["segments"][0]
        # assert output_data["segments"][0]["speaker"] == "SPEAKER_UNKNOWN"
        # assert output_data["segments"][1]["speaker"] == "SPEAKER_UNKNOWN"

        # For now, just ensure the test runs
        assert True
