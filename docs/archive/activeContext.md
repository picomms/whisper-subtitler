# Active Context

## Project Overview
The project is a CLI subtitle generator for local government proceedings videos, using Whisper for transcription and speaker diarization to create accurate subtitles in multiple formats.

## System Information
- **OS**: Ubuntu (24.04)
- **Path Separator**: Forward Slash (/)
- **Python Environment**: Using virtual environment (.venv)

## Current Focus
- Creative design phase complete
- Architecture design: Component-based architecture selected
- Algorithm design: Comprehensive speaker identification pipeline
- Ready for implementation phase

## Technical Stack
- **Python**: Core programming language
- **Whisper**: Transcription engine
- **Pyannote.audio**: Speaker diarization
- **CUDA**: GPU acceleration
- **Output Formats**: TXT, SRT, VTT, TTML/IMSC1
- **Typer**: CLI interface framework
- **Audio Processing**: Librosa and SciPy for preprocessing

## Project Structure
The planned modular architecture:
- Component-based design with clear interfaces
- Configuration system with layered approach (CLI, env, defaults)
- Dedicated modules for transcription, diarization, and output
- Enhanced logging with file and console output
- Comprehensive error handling

## Creative Phase Results
- **Architecture Decision**: Component-based architecture selected for best modularity and testability
- **Algorithm Decision**: Comprehensive speaker identification pipeline with preprocessing, clustering, and optimization
- **Implementation Approach**: Phased implementation with core components first, followed by advanced features
- **Documentation**: Creative phase decisions documented in architecture-design-enhanced.md and algorithm-design-enhanced.md 