# Project Brief: CLI Subtitle Generator for Local Government Proceedings

**Version:** 1.0
**Date:** 2025-05-11

## 1. Introduction & Objective

The primary objective of this project is to develop a robust Command Line Interface (CLI) application that accurately generates subtitles from video files of local government proceedings. These subtitles will subsequently be attached to the video files for delivery via HLS (HTTP Live Streaming). The tool aims to streamline the subtitling process, improve accessibility, and ensure accuracy, particularly in speaker identification.

## 2. Current State

The application is currently operational as a Python script and produces acceptable results. Sensitive information, such as API keys, is managed using a `.env` file.

### 2.1. Important Working Features:
* **Speaker Identification:** Currently functional but identified as a key area for improvement.
* **Multiple File Output:** The script outputs transcriptions and subtitles in several formats to a single designated directory:
    * TXT (Plain text transcript)
    * SRT (SubRip Subtitle format)
    * VTT (WebVTT for HLS compatibility - *Recommendation: Verify if "VTL" in the original brief was a typo for VTT, which is standard for HLS.*)
    * TTML / IMSC1 (Timed Text Markup Language)
* **Full CUDA Support:** GPU acceleration via CUDA is implemented and is a critical feature to preserve for performance.

## 3. Future Plans & Enhancements

### 3.1. Modularization and Code Structure

**Goal:** Refactor the existing script into a modular architecture to enhance maintainability, testability, and scalability for future features.

**Proposed Structure:**

```bash
|- whisper-subtitler/      # Main application package/directory
|  |- modules/             # Core functional modules
|  |  |- __init__.py
|  |  |- transcribe.py    # Handles the core audio-to-text transcription (e.g., Whisper model interaction)
|  |  |- diarise.py       # Manages speaker diarization logic
|  |  |- postprocess.py   # For subtitle formatting, refinement, and future speaker name assignment
|  |  |- output_formats.py # Logic for generating different file formats (SRT, VTT, etc.)
|  |  |- utils.py         # Common utility functions (e.g., file I/O, logging configuration)
|  |- main.py            # Main application logic, orchestrates modules
|  |- __init__.py
|  |- config.py          # Handles loading and management of configuration (from .env, files, CLI)
|- whisperer             # CLI entry point script (e.g., using Click or Typer). Consider making this the installable command.
|- setup.py               # For packaging the application (e.g., for pip installation)
|- requirements.txt       # Lists Python dependencies
|- .env.example           # Example structure for the .env file
|- README.md              # Project overview, installation, quick start, usage
|- tests/                 # Directory for automated tests
|  |- __init__.py
|  |- test_transcribe.py
|  |- test_diarise.py
|  |- test_cli.py
|  |- test_output_formats.py
|  |- fixtures/            # Sample data for tests
```

#### Considerations:

Each module should have clearly defined responsibilities.
Implement robust error handling within and between modules.
Ensure modules can access necessary configurations cleanly (e.g., via a shared config object).

### 3.2. Enhanced Logging
Goal: Implement comprehensive logging, including writing log files to the output directory.

Requirements:

Log files should be created in the specified output directory for each run.
The logging level for the file can be different from the console logging level (e.g., file logs at DEBUG, console at INFO).
Logs should include timestamps, module names, and severity levels.
Consider log rotation for long-term usage to prevent excessively large log files.

### 3.3. Comprehensive Command Line Interface (CLI)

Goal: Develop a user-friendly and powerful CLI using libraries like argparse, Click, or Typer (recommended for its modern features and type hinting).

Desired Command Structure:
whisperer [OPTIONS] --input <VIDEO_FILE_PATH> --output <OUTPUT_DIRECTORY_PATH>

Proposed CLI Arguments & Options:

Required:

-i, --input-file <PATH>: Path to the input video/audio file. (Consider allowing multiple input files or directory input).
-o, --output-dir <PATH>: Path to the directory where output files (transcripts, subtitles, logs) will be saved.
Processing Options:

-l, --language <LANGUAGE_CODE>: Language of the audio content (e.g., en, es). Default to en. Refer to Whisper documentation for supported codes.
-m, --model <MODEL_SIZE>: Whisper model size to use (e.g., tiny, base, small, medium, large, large-v2, large-v3). Default to medium.
--num-speakers <NUMBER>: (Optional) Hint for the number of speakers to aid diarization.
--diarization-model <MODEL_NAME_OR_PATH>: (Future) Specify a particular diarization model if multiple are supported.
--no-cuda: Explicitly disable CUDA usage, even if available.
Output Options:

--output-formats <FORMATS>: Comma-separated list of desired output formats (e.g., srt,vtt,txt). Defaults to all supported (TXT, SRT, VTT, TTML).
-p, --package: If specified, create a TAR archive of all output files (subtitles, transcript, log) in the output directory. The archive should be named based on the input file.
--force-overwrite: Allow overwriting of existing files in the output directory without prompting.
Configuration & API Keys:

--huggingface-token <TOKEN>: Set Hugging Face API token for the current run.
Recommendation: Primarily manage API keys via .env files or environment variables (HUGGINGFACE_TOKEN) for better security. Consider a one-time configuration command like whisperer configure set-hf-token <TOKEN> that saves it to a user-specific config file, rather than passing it frequently via CLI.
Verbosity & Help:

-v, --verbose: Increase console output verbosity (e.g., INFO level).
-vv, --very-verbose: Further increase console output verbosity (e.g., DEBUG level).
-q, --quiet: Suppress all console output except errors.
--version: Display the application version.
-h, --help: Display the help message and exit.

### 3.4. Documentation

Goal: Create comprehensive documentation for users and developers.

Content:

User Documentation:
Installation guide (including CUDA setup if necessary).
Detailed CLI usage instructions with examples for all commands and options.
Supported languages and models.
Troubleshooting common issues.
Developer Documentation:
Code structure overview.
How to contribute (coding standards, testing procedures).
Module API descriptions (if applicable).
Tools: Consider using Sphinx for generating HTML documentation from reStructuredText or Markdown, or simply well-structured Markdown files hosted on a platform like GitHub Pages or ReadTheDocs.

### 3.5. Accuracy Testing Framework

Goal: Establish a framework for systematically testing and tracking transcription and diarization accuracy.

Approach:

Metrics:
Word Error Rate (WER) for transcription.
Diarization Error Rate (DER) for speaker identification.
Dataset: Curate a representative dataset of local government video proceedings with "gold standard" human-verified transcripts and speaker labels.
Methodology: Develop scripts to automate the comparison of generated subtitles against the gold standard.
Tracking: Track accuracy metrics over time and across different models/settings to measure improvements.

### 3.6. Accuracy Improvement Initiatives

Goal: Actively work on improving both transcription and speaker identification accuracy.

Areas of Focus:

Speaker Diarization:
Experiment with advanced diarization models/libraries (e.g., pyannote.audio, NVIDIA NeMo).
Fine-tune diarization models on domain-specific data if possible.
Improve speaker turn detection and assignment.
Transcription (Whisper):
Evaluate different Whisper model sizes and versions for trade-offs between speed and accuracy.
Investigate Whisper parameters (e.g., beam size, temperature, prompt for common jargon/names).
Consider fine-tuning a Whisper model on local government meeting audio if significant domain-specific vocabulary or accents are problematic (a substantial undertaking).
Audio Preprocessing: Implement or enhance audio preprocessing steps:
Noise reduction.
Volume normalization.
Audio channel selection/mixing.
Postprocessing of Transcripts:
Develop rules for correcting common ASR errors specific to the domain (e.g., acronyms, place names, councillor names).
Number formatting, punctuation refinement.

### 3.7. Learn Speaker Names (Very Long-Term Vision)

Goal (Not in immediate scope): Enable the system to learn and associate specific speaker voice characteristics with their names.

Potential Approach: This would likely involve speaker recognition techniques, requiring:

Voice enrollment process for known speakers.
Matching diarized speaker segments against enrolled voice profiles.
A mechanism to manage and update speaker profiles.

## 4. Key Considerations & Recommendations

Configuration Management: Beyond .env and CLI args, consider a dedicated configuration file (e.g., YAML, TOML) for more complex settings (e.g., model paths, postprocessing rules). The config.py module should handle loading from multiple sources (defaults, file, env vars, CLI) in a defined order of precedence.
Error Handling & Resilience: Implement robust error handling throughout the application. Provide clear error messages to the user. The application should handle common issues gracefully (e.g., file not found, invalid audio, CUDA errors, network issues if fetching models).

Testing Strategy:

Unit Tests: For individual functions and classes within modules.

Integration Tests: To ensure modules work together correctly.

CLI Tests: To verify command-line argument parsing and application flow.

Dependencies: Clearly list all external dependencies in requirements.txt, including specific versions where necessary (e.g., openai-whisper, ffmpeg (system dependency), CUDA toolkit version).

Installation & Packaging: Plan for how the CLI tool will be packaged (e.g., using setup.py to create a wheel) and distributed (e.g., PyPI, internal artifact repository). Consider tools like PyInstaller or Nuitka if a standalone executable is desired.

Version Control: Utilize Git for version control. Adopt a branching strategy (e.g., GitFlow, GitHub Flow) if multiple developers are involved.

Licensing: Choose and specify an appropriate open-source or proprietary license for the project.

Performance Monitoring: Beyond CUDA, consider ways to profile and optimize different stages of the pipeline, especially file I/O, audio loading, and model inference.
