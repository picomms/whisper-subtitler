# API Reference

This document provides a detailed reference of the whisper-subtitler API for developers who want to understand the internals or use the application as a library.

## Module Structure

The application is organized into these main modules:

```
whisper-subtitler/
├── modules/
│   ├── audio/          # Audio extraction
│   │   ├── __init__.py
│   │   └── extractor.py
│   ├── config.py       # Configuration management
│   ├── diarisation/    # Speaker identification
│   │   ├── __init__.py
│   │   └── diarizer.py
│   ├── logger.py       # Logging utilities
│   ├── output/         # Output formatting
│   │   ├── __init__.py
│   │   ├── formatter.py
│   │   └── formats.py
│   ├── transcribe/     # Speech transcription
│   │   ├── __init__.py
│   │   └── transcriber.py
│   └── cli.py          # Command-line interface
└── main.py             # Entry point
```

## Core Classes

### Application

The `Application` class in `modules/application.py` orchestrates the entire process.

```python
from modules.config import Config
from modules.application import Application

# Create a configuration
config = Config()
config.input_file = "video.mp4"
config.model_size = "medium"

# Create and run the application
app = Application(config)
output_files = app.process()
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with a configuration object |
| `initialize()` | Initialize all components |
| `process()` | Process the input file and generate outputs |

### Config

The `Config` class in `modules/config.py` manages application configuration.

```python
from modules.config import Config

# Create a configuration
config = Config()

# Load from environment variables
config.load_from_env()

# Load from command-line arguments
config.load_from_args({
    "input_file": "video.mp4",
    "model_size": "small",
    "skip_diarization": True
})

# Validate configuration
config.validate()
```

#### Methods

| Method | Description |
|--------|-------------|
| `load_from_env(env_file=None)` | Load config from .env file |
| `load_from_file(config_file)` | Load config from configuration file |
| `load_from_args(args)` | Load config from command-line arguments |
| `validate()` | Validate configuration values |
| `get(key, default=None)` | Get configuration value with default |

#### Properties

| Property | Type | Default | Description |
|----------|------|---------|-------------|
| `model_size` | str | "base" | Whisper model size |
| `language` | str \| None | "en" | Language code or None for auto |
| `num_speakers` | int \| None | None | Number of speakers or None for auto |
| `huggingface_token` | str \| None | None | HuggingFace API token |
| `skip_diarization` | bool | False | Skip speaker diarization |
| `cluster_speakers` | bool | False | Enable speaker clustering |
| `optimize_num_speakers` | bool | False | Optimize speaker count |
| `preprocess_audio` | bool | False | Preprocess audio for diarization |
| `use_cuda` | bool | True | Use CUDA for GPU acceleration |
| `output_formats` | list[str] | ["txt", "srt", "vtt", "ttml"] | Output formats to generate |
| `input_file` | str \| None | None | Input video file path |
| `output_dir` | str \| None | None | Output directory path |
| `force_overwrite` | bool | False | Force overwrite existing files |
| `audio_sample_rate` | str | "16000" | Audio sample rate |
| `audio_channels` | str | "1" | Audio channels (1=mono, 2=stereo) |
| `verbose` | bool | False | Enable verbose output |
| `log_level` | str | "INFO" | Logging level |
| `log_file` | str \| None | None | Log file path |

### AudioExtractor

The `AudioExtractor` class in `modules/audio/extractor.py` handles audio extraction from video files.

```python
from modules.config import Config
from modules.audio import AudioExtractor

config = Config()
extractor = AudioExtractor(config)
audio_file = extractor.extract_audio("video.mp4", "audio.wav")
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with a configuration object |
| `extract_audio(input_file, output_file=None)` | Extract audio from video file |

### Transcriber

The `Transcriber` class in `modules/transcribe/transcriber.py` handles speech-to-text conversion.

```python
from modules.config import Config
from modules.transcribe import Transcriber

config = Config()
transcriber = Transcriber(config)
result = transcriber.transcribe("audio.wav")
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with a configuration object |
| `transcribe(audio_file)` | Transcribe audio file |
| `get_model()` | Get or load the Whisper model |

### Diarizer

The `Diarizer` class in `modules/diarisation/diarizer.py` handles speaker identification.

```python
from modules.config import Config
from modules.diarisation import Diarizer

config = Config()
config.huggingface_token = "hf_token"
diarizer = Diarizer(config)
speaker_segments = diarizer.diarize("audio.wav")
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with a configuration object |
| `initialize_pipeline()` | Initialize and configure diarization pipeline |
| `diarize(audio_path)` | Identify speakers in audio file |
| `preprocess_audio_file(audio_path)` | Preprocess audio for better diarization |
| `assign_speakers_to_segments(transcription, speaker_segments)` | Combine transcription with speaker info |

### OutputFormatter

The `OutputFormatter` class in `modules/output/formatter.py` manages output generation in different formats.

```python
from modules.config import Config
from modules.output import OutputFormatter

config = Config()
config.output_formats = ["srt", "vtt"]
formatter = OutputFormatter(config)
output_files = formatter.generate_outputs(transcription, "output_base_name")
```

#### Methods

| Method | Description |
|--------|-------------|
| `__init__(config)` | Initialize with a configuration object |
| `generate_outputs(result, base_name=None)` | Generate all configured output formats |

## Format-Specific Classes

The `modules/output/formats.py` file contains classes for each supported output format:

### TXTFormatter

Generates plain text output.

```python
from modules.output.formats import TXTFormatter

formatter = TXTFormatter(config)
formatter.generate(transcription, "output.txt")
```

### SRTFormatter

Generates SubRip Subtitle format.

```python
from modules.output.formats import SRTFormatter

formatter = SRTFormatter(config)
formatter.generate(transcription, "output.srt")
```

### VTTFormatter

Generates WebVTT format.

```python
from modules.output.formats import VTTFormatter

formatter = VTTFormatter(config)
formatter.generate(transcription, "output.vtt")
```

### TTMLFormatter

Generates Timed Text Markup Language format.

```python
from modules.output.formats import TTMLFormatter

formatter = TTMLFormatter(config)
formatter.generate(transcription, "output.ttml")
```

## Logger Module

The `logger.py` module provides logging functionality.

```python
from modules.logger import setup_logging, get_logger

# Set up logging with a configuration
logger = setup_logging(config)

# Get a logger for a specific component
audio_logger = get_logger("audio")
audio_logger.info("Processing audio file")
```

#### Functions

| Function | Description |
|----------|-------------|
| `setup_logging(config)` | Set up logging with configuration |
| `get_logger(name)` | Get a logger for a specific component |

## Using as a Library

While whisper-subtitler is primarily a command-line tool, you can use it as a library in your Python code:

```python
from modules.config import Config
from modules.application import Application

def transcribe_video(video_path, output_dir=None, model="medium", skip_diarization=False):
    """
    Transcribe a video file and generate subtitles.
    
    Args:
        video_path: Path to the video file
        output_dir: Output directory (default: same as video)
        model: Whisper model size (default: medium)
        skip_diarization: Whether to skip speaker identification
        
    Returns:
        Dictionary of output file paths by format
    """
    # Create and configure
    config = Config()
    config.input_file = video_path
    config.output_dir = output_dir
    config.model_size = model
    config.skip_diarization = skip_diarization
    
    # Validate
    config.validate()
    
    # Process
    app = Application(config)
    return app.process()

# Example usage
output_files = transcribe_video("my_video.mp4", model="small", skip_diarization=True)
print(f"Generated SRT: {output_files['srt']}")
```

## Data Structures

### Transcription Result

The result from the Transcriber has this structure:

```python
{
    "text": "Full transcript text...",
    "segments": [
        {
            "id": 0,
            "start": 0.0,
            "end": 3.4,
            "text": "This is the first segment.",
            "speaker": "SPEAKER_01"  # Added by diarization
        },
        {
            "id": 1,
            "start": 3.6,
            "end": 7.2,
            "text": "This is the second segment.",
            "speaker": "SPEAKER_02"  # Added by diarization
        },
        # ... more segments
    ],
    "language": "en"
}
```

### Speaker Segment

The speaker segments from the Diarizer have this structure:

```python
[
    {
        "start": 0.0,
        "end": 3.4,
        "speaker": "SPEAKER_01"
    },
    {
        "start": 3.6,
        "end": 7.2,
        "speaker": "SPEAKER_02"
    },
    # ... more segments
]
``` 