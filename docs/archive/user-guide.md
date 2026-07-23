# User Guide

This guide provides detailed information about using whisper-subtitler for transcription and diarization of video files.

## Table of Contents

1. [Overview](#overview)
2. [Command-Line Interface](#command-line-interface)
3. [Basic Usage](#basic-usage)
4. [Configuration Options](#configuration-options)
5. [Output Formats](#output-formats)
6. [Advanced Features](#advanced-features)
7. [Performance Optimization](#performance-optimization)
8. [Use Cases and Examples](#use-cases-and-examples)

## Overview

whisper-subtitler is a tool for automatically transcribing speech in video files and identifying different speakers. It combines OpenAI's Whisper for transcription and Pyannote.audio for speaker diarization (identifying who is speaking when).

Key features:
- High-quality transcription in multiple languages
- Speaker identification
- Multiple subtitle format outputs (TXT, SRT, VTT, TTML)
- GPU acceleration for faster processing
- Configurable through command-line options or environment variables

## Command-Line Interface

The main command for processing videos is `transcribe`:

```bash
python run.py transcribe [options] input_file
```

To see all available options:

```bash
python run.py transcribe --help
```

### Common Options

| Option             | Description                                           |
| ------------------ | ----------------------------------------------------- |
| `input_file`       | Path to video file (required)                         |
| `-o, --output`     | Output directory (defaults to input file directory)   |
| `-m, --model`      | Whisper model size (tiny, base, small, medium, large) |
| `-l, --language`   | Language code (e.g., "en", "fr", "auto")              |
| `-s, --speakers`   | Number of speakers (if known)                         |
| `-f, --format`     | Output formats to generate                            |
| `-v, --verbose`    | Enable verbose output                                 |
| `--no-diarization` | Skip speaker identification                           |
| `--force`          | Overwrite existing output files                       |
| `--token`          | HuggingFace token for speaker diarization             |
| `--env-file`       | Path to custom .env file                              |

## Basic Usage

### Simplest Usage

The simplest way to use whisper-subtitler is:

```bash
python run.py transcribe video.mp4
```

This will:
1. Load settings from your .env file
2. Extract audio from the video
3. Transcribe the audio using the specified Whisper model 
4. Identify speakers using Pyannote.audio (if not disabled)
5. Generate subtitle files in all supported formats in the same directory as the video

### Without Speaker Identification

If you don't need speaker identification or don't have a HuggingFace token:

```bash
python run.py transcribe video.mp4 --no-diarization
```

This is faster and doesn't require a HuggingFace token.

### Specifying Output Directory

To save output files to a specific directory:

```bash
python run.py transcribe video.mp4 -o /path/to/output
```

This will create all subtitle files in the specified directory.

### Selecting Output Formats

To generate only specific output formats:

```bash
python run.py transcribe video.mp4 -f srt vtt
```

Options include: `txt`, `srt`, `vtt`, `ttml`, or `all`.

## Configuration Options

### Model Selection

Whisper offers several model sizes with different accuracy and speed trade-offs:

```bash
python run.py transcribe video.mp4 -m small
```

| Model  | Size  | Accuracy | Speed   | Memory Usage |
| ------ | ----- | -------- | ------- | ------------ |
| tiny   | 39M   | Lowest   | Fastest | Lowest       |
| base   | 74M   | Low      | Fast    | Low          |
| small  | 244M  | Medium   | Medium  | Medium       |
| medium | 769M  | High     | Slow    | High         |
| large  | 1.5GB | Highest  | Slowest | Highest      |

### Language Selection

By default, whisper-subtitler uses English. To specify another language:

```bash
python run.py transcribe video.mp4 -l fr
```

For auto-detection of language:

```bash
python run.py transcribe video.mp4 -l auto
```

### Environment Variables

Instead of passing command-line options each time, you can configure defaults in a `.env` file:

```
# .env file example
WHISPER_MODEL_SIZE=medium
WHISPER_LANGUAGE=en
HUGGINGFACE_TOKEN=hf_your_token_here
SKIP_DIARIZATION=false
```

To use a custom environment file:

```bash
python run.py transcribe video.mp4 --env-file .env.custom
```

## Output Formats

whisper-subtitler supports multiple subtitle formats:

### TXT (Plain Text)

Simple text transcript without timing information:

```
This is a transcript of the spoken content in the video.
It does not contain any timestamps or speaker identification.
```

### SRT (SubRip)

The most common subtitle format with sequence numbers, timecodes, and text:

```
1
00:00:00,000 --> 00:00:03,400
Speaker 1: This is the first subtitle.

2
00:00:03,600 --> 00:00:07,200
Speaker 2: This is the second subtitle.
```

### VTT (WebVTT)

Web Video Text Tracks format, primarily used for web videos:

```
WEBVTT

00:00:00.000 --> 00:00:03.400
Speaker 1: This is the first subtitle.

00:00:03.600 --> 00:00:07.200
Speaker 2: This is the second subtitle.
```

### TTML (Timed Text Markup Language)

XML-based format used by professional broadcast systems:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<tt xml:lang="en-GB" xmlns="http://www.w3.org/ns/ttml">
  <head>
    <metadata>
      <ttm:title>Transcription</ttm:title>
    </metadata>
    <!-- styling and layout elements -->
  </head>
  <body>
    <div>
      <p begin="00:00:00.000" end="00:00:03.400">This is the first subtitle.</p>
      <p begin="00:00:03.600" end="00:00:07.200">This is the second subtitle.</p>
    </div>
  </body>
</tt>
```

## Advanced Features

### Transcription Accuracy Improvements

#### Whisper Model Parameters

Fine-tune the Whisper transcription with these parameters:

```bash
# Adjust beam search settings for better accuracy
python run.py transcribe video.mp4 --beam-size 8 --best-of 8

# Lower temperature for more deterministic output (good for clear audio)
python run.py transcribe video.mp4 --temperature 0.0

# Higher temperature for more creative output (may help with difficult audio)
python run.py transcribe video.mp4 --temperature 0.2

# Initial prompt to guide transcription (useful for domain-specific terminology)
python run.py transcribe video.mp4 --initial-prompt "Meeting transcript with John and Sarah discussing project deadlines."
```

#### Auto Model Selection

Let the application automatically select the optimal Whisper model based on your audio:

```bash
# Auto-select best model with balanced accuracy/speed criteria
python run.py transcribe video.mp4 --auto-model

# Auto-select optimizing for accuracy
python run.py transcribe video.mp4 --auto-model --model-criteria accuracy

# Auto-select optimizing for speed
python run.py transcribe video.mp4 --auto-model --model-criteria speed
```

#### Audio-Optimized Transcription

Automatically optimize transcription parameters based on audio characteristics:

```bash
python run.py transcribe video.mp4 --optimize-audio
```

This analyzes your audio to detect characteristics like noise level, speech density, etc., and adjusts Whisper parameters accordingly.

#### Audio Preprocessing

Improve transcription quality with audio preprocessing:

```bash
# Apply volume normalization
python run.py transcribe video.mp4 --normalize-audio

# Apply noise reduction
python run.py transcribe video.mp4 --reduce-noise

# Apply high-pass filter to reduce low-frequency noise
python run.py transcribe video.mp4 --high-pass --cutoff 100
```

You can combine multiple preprocessing options:

```bash
python run.py transcribe video.mp4 --normalize-audio --reduce-noise --high-pass
```

### Speaker Diarization Options

#### Known Number of Speakers

If you know how many speakers are in your video:

```bash
python run.py transcribe video.mp4 --speakers 2
```

This improves accuracy by giving the diarization system the exact number of speakers to identify.

You can also specify a range:

```bash
python run.py transcribe video.mp4 --min-speakers 2 --max-speakers 4
```

#### Speaker Clustering

For better consistency in speaker labels:

```bash
python run.py transcribe video.mp4 --cluster
```

This helps when the same speaker may be initially identified as different speakers due to changes in voice tone or audio quality.

You can adjust the similarity threshold for clustering:

```bash
python run.py transcribe video.mp4 --cluster --similarity-threshold 0.75
```
Lower values merge speakers more aggressively, higher values require more similarity to consider speakers the same.

#### Speaker Optimization

When you know the number of speakers and want to improve identification:

```bash
python run.py transcribe video.mp4 --speakers 3 --optimize-speakers
```

#### Voice Activity Detection

Improve speaker segmentation by accurately detecting when people are speaking:

```bash
python run.py transcribe video.mp4 --voice-activity --min-silence 0.5
```

The `min-silence` parameter (in seconds) controls how long a silence must be to consider it a break between speech segments.

#### Audio Preprocessing

For better diarization in challenging audio conditions:

```bash
python run.py transcribe video.mp4 --preprocess
```

This applies noise reduction and audio filtering before diarization.

### Overwriting Existing Outputs

By default, the application won't overwrite existing output files. To force overwriting:

```bash
python run.py transcribe video.mp4 --force
```

## Performance Optimization

### GPU Acceleration

By default, whisper-subtitler uses GPU if available. To force CPU mode:

```bash
python run.py transcribe video.mp4 --cpu
```

### Model Size Selection

For faster processing with slightly lower accuracy:

```bash
python run.py transcribe video.mp4 -m small
```

For highest accuracy but slower processing:

```bash
python run.py transcribe video.mp4 -m large
```

### Memory Usage

The memory requirements depend primarily on the model size:

| Model Size | Approximate RAM Required |
|------------|--------------------------|
| tiny       | 1 GB                     |
| base       | 1 GB                     |
| small      | 2 GB                     |
| medium     | 5 GB                     |
| large      | 10 GB                    |

If you encounter memory issues, try a smaller model.

## Use Cases and Examples

### Quick Transcription Without Speaker Identification

For rapid transcription without caring about who is speaking:

```bash
python run.py transcribe video.mp4 -m small --no-diarization
```

### High-Quality Podcast Transcription

For podcasts with 2-3 speakers where accuracy is important:

```bash
python run.py transcribe podcast.mp4 -m large --speakers 3 --cluster --optimize-audio
```

### Foreign Language Video

For a video in French with multiple speakers:

```bash
python run.py transcribe french_video.mp4 -l fr --cluster
```

### Noisy Audio Source

For videos with background noise or low-quality audio:

```bash
python run.py transcribe noisy_video.mp4 --normalize-audio --reduce-noise --high-pass
```

### Technical Content with Specialized Terminology

For technical videos with domain-specific terminology:

```bash
python run.py transcribe technical_video.mp4 --initial-prompt "Technical discussion about machine learning with terms like CNN, RNN, LSTM, and transformer architecture."
```

### Batch Processing

To process multiple files with the same settings:

```bash
# Bash example
for video in *.mp4; do
  python run.py transcribe "$video" --no-diarization
done
```

### Using a Custom Configuration

For recurring tasks with specific settings:

1. Create a custom .env file:

```
# .env.podcasts
WHISPER_MODEL_SIZE=medium
WHISPER_LANGUAGE=en
NUM_SPEAKERS=2
CLUSTER_SPEAKERS=true
OPTIMIZE_NUM_SPEAKERS=true
OPTIMIZE_FOR_AUDIO=true
```

2. Use it for transcription:

```bash
python run.py transcribe podcast.mp4 --env-file .env.podcasts
``` 