# Whisper-Subtitler Architecture Design

## 🏗️ High-Level Architecture

```
whisper-subtitler/
├── modules/               # Core functional modules
│   ├── __init__.py
│   ├── transcribe.py      # Handles Whisper model transcription
│   ├── diarise.py         # Manages speaker diarization
│   ├── postprocess.py     # Subtitle formatting and refinement
│   ├── output_formats.py  # Generation of different file formats
│   └── utils.py           # Common utility functions
├── main.py                # Main application orchestration logic
├── __init__.py
├── config.py              # Configuration management
└── cli.py                 # CLI interface implementation

whisperer                  # CLI entry point script
```

## 📦 Component Design

### 1. Configuration Management (`config.py`)

**Responsibility**: Manage all application configuration from multiple sources.

**Design**:
- Load configuration from multiple sources in this order of precedence:
  1. Command-line arguments (highest priority)
  2. Environment variables (.env file)
  3. Default values (lowest priority)
- Use a centralized Config class that other modules can import
- Support validation of configuration values
- Expose configuration with proper typing

```python
class Config:
    def __init__(self):
        # Default values
        self.model_size = "medium"
        self.language = "en"
        self.output_formats = ["txt", "srt", "vtt", "ttml"]
        self.num_speakers = None
        # ...other config values
        
    def load_from_env(self):
        # Load from environment variables
        
    def load_from_args(self, args):
        # Load from CLI args
        
    def validate(self):
        # Validate configuration
```

### 2. Transcription Module (`modules/transcribe.py`)

**Responsibility**: Handle audio transcription using Whisper.

**Design**:
- Encapsulate Whisper model initialization
- Support different model sizes
- Support language selection
- Handle CUDA/CPU device selection
- Return structured transcription results

```python
class Transcriber:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def load_model(self):
        # Initialize and load the Whisper model
        
    def transcribe(self, audio_file):
        # Transcribe the audio file
        # Return structured results
```

### 3. Speaker Diarization (`modules/diarise.py`)

**Responsibility**: Identify different speakers in the audio.

**Design**:
- Encapsulate Pyannote.audio initialization
- Support speaker count parameter
- Handle authentication with HuggingFace
- Support CUDA acceleration
- Return structured speaker segments

```python
class Diarizer:
    def __init__(self, config):
        self.config = config
        self.pipeline = None
        
    def load_pipeline(self):
        # Initialize diarization pipeline
        
    def diarize(self, audio_file):
        # Run speaker diarization
        # Return structured speaker segments
```

### 4. Output Format Generation (`modules/output_formats.py`)

**Responsibility**: Generate different subtitle formats.

**Design**:
- Use a factory pattern for different output formats
- Common interface for all format generators
- Support selective format generation
- Ensure consistent timing and formatting

```python
class OutputFormatter:
    def __init__(self, config):
        self.config = config
        
    def get_formatter(self, format_type):
        # Return the appropriate formatter based on type
        
class BaseFormatter:
    def generate(self, transcription, speakers, output_path):
        # Abstract method to be implemented by subclasses
        
class SRTFormatter(BaseFormatter):
    def generate(self, transcription, speakers, output_path):
        # Generate SRT format
        
class VTTFormatter(BaseFormatter):
    def generate(self, transcription, speakers, output_path):
        # Generate VTT format
        
# Other formatters...
```

### 5. Logging System (`modules/utils.py`)

**Responsibility**: Provide comprehensive logging functionality.

**Design**:
- Configure both console and file logging
- Support different log levels for each
- Create log files in output directory
- Include timestamps and module information

```python
def setup_logging(config):
    # Configure logging based on config
    # Set up console handler
    # Set up file handler if output directory specified
    # Set log levels based on verbosity
```

### 6. CLI Interface (`cli.py` and `whisperer`)

**Responsibility**: Provide user-friendly command-line interface.

**Design**:
- Use Typer for modern CLI with type hints
- Organize commands logically
- Provide helpful error messages
- Support all required options from project brief
- Handle validation of command-line arguments

```python
import typer

app = typer.Typer()

@app.command()
def transcribe(
    input_file: str = typer.Option(..., "--input", "-i", help="Input video file"),
    output_dir: str = typer.Option(..., "--output", "-o", help="Output directory"),
    # ...other options
):
    """Transcribe video and generate subtitles."""
    # Implementation
```

### 7. Main Application Logic (`main.py`)

**Responsibility**: Orchestrate the overall transcription process.

**Design**:
- Initialize configuration
- Set up logging
- Create instances of required components
- Handle the overall workflow
- Manage error handling and reporting

```python
class Application:
    def __init__(self, config):
        self.config = config
        self.transcriber = None
        self.diarizer = None
        self.formatter = None
        
    def initialize(self):
        # Initialize components
        
    def run(self):
        # Run the complete workflow
        # Handle errors and report status
```

## 🔄 Data Flow

1. User invokes CLI command with arguments
2. CLI parses arguments and initializes configuration
3. Main application creates and initializes components
4. Audio extraction from video (if needed)
5. Speaker diarization processes audio
6. Whisper transcription processes audio
7. Results are combined and post-processed
8. Output formatters generate selected formats
9. Results are saved to the output directory
10. Summary and logs are displayed to user

## 🔒 Error Handling Strategy

- Each module will handle its own domain-specific errors
- All errors will be logged appropriately
- User-friendly error messages will be displayed
- Critical errors will halt execution with appropriate exit codes
- Non-critical errors will be reported but allow continued execution

## 🧪 Testability Considerations

- Dependencies will be injected rather than created directly
- Modules will have clear interfaces for mocking
- Configuration will be centralized for easier testing
- Functions will be focused on single responsibilities
- External services (Whisper, Pyannote) will be abstracted

## 🚀 Future Extensibility

The architecture is designed to support future enhancements:

- Additional output formats can be added by creating new formatters
- Speaker name learning could be implemented in a new module
- Audio preprocessing could be added as a pipeline stage
- Additional transcription models could be supported through abstraction
- Custom post-processing rules can be implemented in the postprocess module 