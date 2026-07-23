# API Design for Callable Utility

🎨🎨🎨 **ENTERING CREATIVE PHASE: ARCHITECTURE DESIGN**

## Component Description

The API component will transform whisper-subtitler from a command-line tool into a callable Python library that can be easily integrated into other Python applications. The core functionality will be exposed through a clean, well-documented API that maintains all the existing features while providing a programmatic interface.

## Requirements & Constraints

### Functional Requirements

1. Provide a callable `whisper_subtitler()` function as the main entry point
2. Support all existing configuration options from the CLI
3. Return structured results including output file paths and transcription data
4. Support progress reporting through callbacks
5. Enable programmatic access to transcription and diarization results
6. Maintain the same accuracy and performance as the CLI version

### Non-Functional Requirements

1. Provide clear error messages and proper exception handling
2. Maintain backward compatibility with existing code
3. Create comprehensive documentation and examples
4. Support proper type hints for IDE integration
5. Minimize dependencies on user's environment

### Constraints

1. Must work with the existing Application class
2. Should not duplicate code between CLI and API interfaces
3. Should handle resource management properly
4. Must maintain the existing configuration system

## Multiple Options Analysis

### Option 1: Thin Wrapper Approach

**Description:**
Create a simple wrapper function around the existing Application class with minimal changes to the core code.

**Architecture:**
```
whisper_subtitler()
└── Creates Config object
    └── Instantiates Application
        └── Calls app.process()
            └── Returns output paths and transcript data
```

**Pros:**
- Simple implementation with minimal changes
- Maintains full compatibility with existing code
- Lowest risk of introducing bugs

**Cons:**
- Limited flexibility for API-specific features
- Difficult to implement progress reporting
- May include unnecessary file I/O operations
- Tied closely to current implementation details

### Option 2: Complete API Redesign

**Description:**
Create a new, separate API layer with its own abstractions specifically designed for programmatic use.

**Architecture:**
```
whisper_subtitler()
└── API Layer
    ├── TranscriptionManager
    ├── DiarizationManager
    └── OutputManager
```

**Pros:**
- Clean API design optimized for programmatic use
- Can implement optimal progress reporting
- More flexibility for future API enhancements
- Better separation from CLI implementation

**Cons:**
- Significant duplication of functionality
- Higher risk of diverging implementations
- More complex to maintain
- Larger development effort required

### Option 3: Shared Core with API Adaptation

**Description:**
Refactor the Application class to support both CLI and API use cases, with an adapter layer for the API.

**Architecture:**
```
whisper_subtitler()
└── ApiAdapter
    └── Application (shared core)
        ├── Configurable output options
        ├── Progress callback support
        ├── Resource management
        └── Returns structured results
```

**Pros:**
- Reuses existing code without duplication
- Maintains consistency between CLI and API
- Can add API-specific features where needed
- Moderate development effort
- Easier to maintain in the long term

**Cons:**
- Requires some refactoring of existing code
- Slightly more complex than a thin wrapper
- May require balancing CLI and API needs in shared code

## Recommended Approach

After analyzing the options, **Option 3: Shared Core with API Adaptation** provides the best balance of code reuse, flexibility, and maintainability.

### Justification:

1. **Code Reuse**: Avoids duplication by leveraging the existing Application class.
2. **Consistency**: Ensures API and CLI behaviors remain consistent.
3. **Flexibility**: Allows for API-specific enhancements where needed.
4. **Maintainability**: Changes to core functionality automatically benefit both interfaces.
5. **Development Effort**: Moderate effort compared to a complete redesign.

Option 1 is too limited for a proper API experience, while Option 2 would create unnecessary duplication and maintenance burden. Option 3 strikes the right balance by adapting the existing architecture for API use while maintaining a unified codebase.

## Implementation Guidelines

### 1. API Module

Create a new `api.py` module with the main entry point function:

```python
"""
API module for Whisper-Subtitler.

This module provides a clean API for using Whisper-Subtitler
as a library in other Python applications.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from .modules import Application, Config
from .modules.logger import get_logger

logger = get_logger("api")

def whisper_subtitler(
    input_file: str,
    output_dir: Optional[str] = None,
    model_size: str = "base",
    language: Optional[str] = "en",
    output_formats: List[str] = ["txt", "srt", "vtt", "ttml"],
    num_speakers: Optional[int] = None,
    gpu_backend: str = "auto",
    skip_diarization: bool = False,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    huggingface_token: Optional[str] = None,
    return_transcription: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Transcribe audio/video and identify speakers.
    
    Args:
        input_file: Path to input audio/video file
        output_dir: Directory to save output files (defaults to input file directory)
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code or None for auto-detection
        output_formats: List of output formats to generate
        num_speakers: Number of speakers or None for auto-detection
        gpu_backend: GPU backend to use (auto, cuda, rocm, mps, cpu)
        skip_diarization: Skip speaker diarization
        progress_callback: Callback function for progress updates
        huggingface_token: Hugging Face API token for PyAnnote
        return_transcription: Whether to include full transcription data in result
        **kwargs: Additional configuration options
        
    Returns:
        Dictionary containing:
        - output_files: Dictionary of output files by format
        - transcription: Complete transcription data (if return_transcription=True)
        - segments: List of segments with timing and speaker info
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If configuration options are invalid
        RuntimeError: For processing errors
    """
    # Input validation
    if not input_file or not Path(input_file).exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    # Create configuration
    config = Config()
    
    # Set basic configuration
    config.input_file = input_file
    config.output_dir = output_dir
    config.model_size = model_size
    config.language = language
    config.output_formats = output_formats
    config.num_speakers = num_speakers
    config.gpu_backend = gpu_backend
    config.skip_diarization = skip_diarization
    config.huggingface_token = huggingface_token
    
    # Set log level to warning by default for API usage unless specified
    if "log_level" not in kwargs:
        config.log_level = "WARNING"
    
    # Set additional configuration from kwargs
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    try:
        # Create and initialize application
        app = Application(config)
        
        # Register progress callback if provided
        if progress_callback:
            app.register_progress_callback(progress_callback)
        
        # Run the application
        output_files = app.process()
        
        # Build result dictionary
        result = {
            "output_files": output_files,
            "segments": app.transcriber.last_transcription["segments"] if app.transcriber else []
        }
        
        # Include full transcription data if requested
        if return_transcription and app.transcriber:
            result["transcription"] = app.transcriber.last_transcription
        
        return result
        
    except Exception as e:
        logger.error(f"Error in whisper_subtitler: {e}")
        # Re-raise with context
        if isinstance(e, (FileNotFoundError, ValueError)):
            raise
        else:
            raise RuntimeError(f"Transcription failed: {e}") from e
```

### 2. Application Class Updates

Enhance the Application class to support API usage, particularly for progress reporting:

```python
# In application.py

class Application:
    """Main application orchestrator."""
    
    def __init__(self, config):
        # ... existing code ...
        self.progress_callbacks = []
        self.last_progress = 0
    
    def register_progress_callback(self, callback):
        """Register a progress callback function.
        
        Args:
            callback: Function with signature (stage: str, progress: float)
        """
        if callable(callback):
            self.progress_callbacks.append(callback)
    
    def report_progress(self, stage, progress):
        """Report progress to all registered callbacks.
        
        Args:
            stage: Current processing stage
            progress: Progress value between 0 and 1
        """
        if self.progress_callbacks:
            for callback in self.progress_callbacks:
                try:
                    callback(stage, progress)
                except Exception as e:
                    self.logger.warning(f"Error in progress callback: {e}")
        
        self.last_progress = progress
    
    def process(self):
        """Process the input file with all components."""
        try:
            # ... existing code ...
            
            # Report progress at various stages
            self.report_progress("Initializing", 0.1)
            
            # Extract audio from input file
            # ... existing code ...
            self.report_progress("Extracting audio", 0.2)
            
            # Run transcription
            # ... existing code ...
            self.report_progress("Transcribing", 0.5)
            
            # Run speaker diarization if not skipped
            # ... existing code ...
            self.report_progress("Diarizing", 0.8)
            
            # Generate output files
            # ... existing code ...
            self.report_progress("Generating output", 0.9)
            
            # ... existing code ...
            self.report_progress("Complete", 1.0)
            
            return output_files
            
        except Exception as e:
            self.report_progress("Error", 1.0)
            raise
```

### 3. Transcriber Class Updates

Update the Transcriber class to store the last transcription result:

```python
# In transcriber.py

class Transcriber:
    def __init__(self, config):
        # ... existing code ...
        self.last_transcription = None
    
    def transcribe(self, audio_path, reference_text=None):
        # ... existing code ...
        
        # Store the transcription result for API access
        self.last_transcription = transcription
        
        return transcription
```

### 4. Update Package Initialization

Update `__init__.py` to expose the API function:

```python
"""
Whisper-Subtitler: Video transcription and diarization.

A tool for transcribing videos and identifying speakers
using OpenAI's Whisper and Pyannote.
"""

from .modules import Application, Config, get_logger, setup_logging
from .version import VERSION
from .api import whisper_subtitler

__version__ = VERSION
__all__ = [
    "Application", 
    "Config", 
    "__version__", 
    "get_logger", 
    "setup_logging",
    "whisper_subtitler"
]
```

### 5. Example Usage

Create an example script to demonstrate API usage:

```python
"""
Example usage of whisper-subtitler as a Python library.
"""

from whisper_subtitler import whisper_subtitler

def progress_callback(stage, progress):
    """Display progress updates."""
    print(f"{stage}: {progress*100:.1f}%")

# Transcribe a video file
result = whisper_subtitler(
    input_file="video.mp4",
    model_size="small",
    num_speakers=2,
    progress_callback=progress_callback
)

# Print output file paths
print("\nOutput files:")
for format, path in result["output_files"].items():
    print(f"{format}: {path}")

# Print first few segments
print("\nFirst 3 segments:")
for segment in result["segments"][:3]:
    print(f"{segment['speaker']} ({segment['start']:.1f}s - {segment['end']:.1f}s): {segment['text']}")
```

## Error Handling Strategy

1. **Input Validation**: Validate all inputs at the API boundary
2. **Specific Exceptions**: Use appropriate exception types for different error cases
3. **Context Preservation**: Maintain original exceptions as causes of raised exceptions
4. **Clean Resource Management**: Ensure resources are properly released, even on errors
5. **Meaningful Messages**: Provide clear error messages that guide toward resolution

## Testing Strategy

1. **Unit Tests**: Test the API function with different parameter combinations
2. **Mock Testing**: Use mocks to test error handling and progress reporting
3. **Integration Tests**: Verify integration with Application class
4. **Example Validation**: Ensure example scripts work as expected

## Documentation Strategy

1. **API Reference**: Create comprehensive API reference documentation
2. **Usage Examples**: Provide multiple examples for different use cases
3. **Parameter Documentation**: Document all parameters with valid values
4. **Type Hints**: Use proper type hints for better IDE integration
5. **Integration Guide**: Create a guide for integrating with existing Python applications

## Verification

This design satisfies all the requirements:

1. ✅ Provides a callable `whisper_subtitler()` function
2. ✅ Supports all existing configuration options
3. ✅ Returns structured results
4. ✅ Implements progress reporting via callbacks
5. ✅ Enables programmatic access to results
6. ✅ Maintains compatibility with existing code
7. ✅ Provides appropriate error handling
8. ✅ Includes documentation and examples

🎨🎨🎨 **EXITING CREATIVE PHASE** 