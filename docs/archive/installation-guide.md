# Installation Guide

This guide will walk you through the process of installing and setting up whisper-subtitler.

## Prerequisites

Before installing whisper-subtitler, make sure you have the following prerequisites:

1. **Python 3.11+**
   - The application requires Python 3.11 or newer
   - Verify your Python version with `python --version`

2. **FFmpeg**
   - Required for extracting audio from video files
   - Installation varies by platform (see platform-specific instructions below)

3. **CUDA** (Optional, but recommended)
   - Required for GPU acceleration
   - Makes transcription and diarization significantly faster
   - Install CUDA 11.7+ if you have a compatible NVIDIA GPU

4. **HuggingFace Token**
   - Required for speaker diarization
   - Not required if using `--no-diarization` option
   - Get your token from [HuggingFace Settings](https://hf.co/settings/tokens)
   - You must accept the license terms at [pyannote/speaker-diarization](https://hf.co/pyannote/speaker-diarization)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/picommcapp/whisper-subtitler.git
cd whisper-subtitler
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create Configuration File

Create a `.env` file in the project root directory:

```bash
# Copy the sample file
cp .env.sample .env

# Edit the file with your preferred text editor
nano .env  # or use any text editor
```

Update the `.env` file with your HuggingFace token and preferred settings.

## Platform-Specific Installation

### Windows

1. **Install FFmpeg**:
   - Download from [FFmpeg Official Website](https://ffmpeg.org/download.html)
   - Add FFmpeg to your PATH environment variable
   - Verify with `ffmpeg -version` in Command Prompt

2. **Install CUDA** (Optional):
   - Download CUDA Toolkit from [NVIDIA's website](https://developer.nvidia.com/cuda-downloads)
   - Follow the installer instructions
   - Make sure to install the cuDNN library as well

### macOS

1. **Install FFmpeg**:
   ```bash
   # Using Homebrew
   brew install ffmpeg
   
   # Verify installation
   ffmpeg -version
   ```

2. **Note on CUDA**: 
   - CUDA is not officially supported on macOS
   - For macOS, the application will automatically use CPU mode

### Linux (Ubuntu/Debian)

1. **Install FFmpeg**:
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ffmpeg -version
   ```

2. **Install CUDA** (Optional):
   - Follow [NVIDIA's official instructions](https://developer.nvidia.com/cuda-downloads)
   ```bash
   # Example installation for Ubuntu 22.04
   wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
   sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
   wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
   sudo dpkg -i cuda-repo-ubuntu2204-12-2-local_12.2.0-535.54.03-1_amd64.deb
   sudo cp /var/cuda-repo-ubuntu2204-12-2-local/cuda-*-keyring.gpg /usr/share/keyrings/
   sudo apt-get update
   sudo apt-get -y install cuda
   ```

## Verifying Installation

To verify that everything is set up correctly:

```bash
# Run the version command
python run.py version

# Test with a sample video file (without diarization)
python run.py transcribe path/to/video.mp4 --no-diarization
```

## Installation Troubleshooting

### Common Issues

1. **FFmpeg not found**
   - Error: `FileNotFoundError: [Errno 2] No such file or directory: 'ffmpeg'`
   - Solution: Make sure FFmpeg is installed and in your PATH

2. **CUDA not detected**
   - Warning: `CUDA is not available, using CPU. This will be much slower!`
   - Solution: Check CUDA installation, or run with `--cpu` flag

3. **HuggingFace Authentication Error**
   - Error: `HuggingFace authentication error` or `401 Client Error`
   - Solution: Check your token or use `--no-diarization` option

4. **Python Version Error**
   - Error: `SyntaxError: ...` or incompatible module errors
   - Solution: Make sure you're using Python 3.9 or newer

### Getting Help

If you encounter any issues not covered here, please:
1. Check the [GitHub Issues](https://github.com/picommcapp/whisper-subtitler/issues)
2. Open a new issue if needed, providing detailed information about your problem 