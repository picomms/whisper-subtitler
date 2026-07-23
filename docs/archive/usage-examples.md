# Usage Examples

This document provides practical examples of how to use whisper-subtitler in different scenarios.

## Basic Usage

### Simple Transcription

The most basic way to use whisper-subtitler is to transcribe a video file:

```bash
python run.py transcribe path/to/video.mp4
```

This will:
1. Extract audio from the video
2. Transcribe the audio using the default Whisper model
3. Perform speaker diarization (if enabled)
4. Generate output in all supported formats (TXT, SRT, VTT, TTML)

### Transcription Without Speaker Identification

If you don't have a HuggingFace token or don't need speaker identification:

```bash
python run.py transcribe path/to/video.mp4 --no-diarization
```

This is faster and doesn't require a HuggingFace token.

### Specify Output Directory

To save the output files in a specific directory:

```bash
python run.py transcribe path/to/video.mp4 -o path/to/output
```

By default, files are saved in the same directory as the input video.

## Advanced Usage

### Using a Specific Model

Choose a Whisper model based on your needs:

```bash
# For high accuracy but slower processing
python run.py transcribe path/to/video.mp4 -m large

# For faster processing with decent accuracy
python run.py transcribe path/to/video.mp4 -m small

# For fastest processing
python run.py transcribe path/to/video.mp4 -m tiny
```

### Setting Language

Specify the language of the audio (improves accuracy):

```bash
# For English content
python run.py transcribe path/to/video.mp4 -l en

# For French content
python run.py transcribe path/to/video.mp4 -l fr

# For auto-detection
python run.py transcribe path/to/video.mp4 -l auto
```

### Improving Speaker Identification

When you know the number of speakers:

```bash
python run.py transcribe path/to/video.mp4 --speakers 2
```

For more consistent speaker labeling:

```bash
python run.py transcribe path/to/video.mp4 --cluster
```

### Custom Environment Settings

Use a custom environment file:

```bash
python run.py transcribe path/to/video.mp4 --env-file .env.custom
```

## Accuracy Improvement Examples

### Automatic Model Selection

Let the system choose the optimal Whisper model for your audio:

```bash
# Balance between accuracy and speed
python run.py transcribe interview.mp4 --auto-model

# Prioritize accuracy
python run.py transcribe important_lecture.mp4 --auto-model --model-criteria accuracy

# Prioritize speed
python run.py transcribe quick_notes.mp4 --auto-model --model-criteria speed
```

### Optimizing for Audio Type

Automatically optimize transcription parameters based on your audio characteristics:

```bash
python run.py transcribe conference_talk.mp4 --optimize-audio
```

### Using Advanced Whisper Parameters

Fine-tune transcription results with Whisper's advanced parameters:

```bash
# Adjust beam search for better accuracy
python run.py transcribe technical_presentation.mp4 --beam-size 8 --best-of 8

# Use initial prompt to guide transcription with domain-specific terms
python run.py transcribe medical_lecture.mp4 --initial-prompt "Medical lecture discussing cardiovascular health, hypertension, and atherosclerosis."
```

### Audio Preprocessing

Improve transcription quality for challenging audio:

```bash
# Normalize audio volume
python run.py transcribe quiet_recording.mp4 --normalize-audio

# Reduce background noise
python run.py transcribe noisy_interview.mp4 --reduce-noise

# Apply high-pass filter to remove rumble
python run.py transcribe outdoor_speech.mp4 --high-pass --cutoff 100

# Combine multiple preprocessing techniques
python run.py transcribe difficult_audio.mp4 --normalize-audio --reduce-noise --high-pass
```

### Enhanced Speaker Diarization

Improve speaker identification in complex scenarios:

```bash
# Specify speaker range when exact count is unknown
python run.py transcribe panel_discussion.mp4 --min-speakers 3 --max-speakers 5

# Adjust speaker clustering threshold for similar voices
python run.py transcribe family_conversation.mp4 --cluster --similarity-threshold 0.8

# Improve speaker segmentation with voice activity detection
python run.py transcribe podcast.mp4 --voice-activity --min-silence 0.4
```

## Specialized Use Cases

### Podcast Transcription

For podcasts with multiple speakers:

```bash
python run.py transcribe podcast.mp4 --speakers 3 --cluster --optimize-audio --initial-prompt "Podcast discussion about technology trends."
```

### Lecture Transcription

For academic lectures with technical terms:

```bash
python run.py transcribe lecture.mp4 -m large --optimize-audio --initial-prompt "Computer science lecture about algorithms and data structures."
```

### Noisy Interview

For interviews in noisy environments:

```bash
python run.py transcribe noisy_interview.mp4 --normalize-audio --reduce-noise --high-pass --preprocess --cluster
```

### Multi-language Content

For content with multiple languages:

```bash
# Let Whisper auto-detect the language
python run.py transcribe multilingual.mp4 -l auto -m large
```

### Batch Processing

Process multiple files with the same settings:

```bash
# Bash script example
for video in *.mp4; do
  echo "Processing $video..."
  python run.py transcribe "$video" --no-diarization --optimize-audio
done
```

## Configuration Files

For recurring use cases, create custom configuration files:

### Example .env.podcast

```
# .env.podcast
WHISPER_MODEL_SIZE=medium
WHISPER_LANGUAGE=en
NUM_SPEAKERS=2
CLUSTER_SPEAKERS=true
OPTIMIZE_NUM_SPEAKERS=true
OPTIMIZE_FOR_AUDIO=true
INITIAL_PROMPT=Podcast discussion between host and guest.
NOISE_REDUCTION=true
AUDIO_NORMALIZATION=true
```

Use it with:

```bash
python run.py transcribe podcast_episode.mp4 --env-file .env.podcast
``` 