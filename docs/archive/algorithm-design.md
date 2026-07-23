# Algorithm Design: Speaker Identification Improvements

## Current Limitations

The current speaker identification approach in `subwhisper.py` has the following limitations:

1. **Simple Association**: Currently assigns speakers based on temporal overlap only
2. **No Voice Characteristics**: Doesn't use voice characteristics for identification
3. **No Speaker Consistency**: Can assign different speaker IDs to the same person 
4. **Limited Configuration**: No way to specify known number of speakers
5. **No Name Assignment**: No mechanism to assign real names to speakers

## Proposed Improvements

### 1. Enhanced Speaker Segment Association

**Current Algorithm:**
```python
def assign_speaker(whisper_segment: dict[str, Any]) -> str:
    for seg in speaker_segments:
        if seg["start"] <= whisper_segment["start"] < seg["end"]:
            return seg["speaker"]
    return "Unknown"
```

**Improved Algorithm:**
```python
def assign_speaker(whisper_segment: dict[str, Any], speaker_segments: list) -> str:
    # First check for direct overlap with segment start
    for seg in speaker_segments:
        if seg["start"] <= whisper_segment["start"] < seg["end"]:
            return seg["speaker"]
    
    # If no match at start point, check for maximum overlap
    max_overlap = 0
    best_speaker = "Unknown"
    segment_start = whisper_segment["start"]
    segment_end = whisper_segment["end"]
    
    for seg in speaker_segments:
        # Calculate overlap duration
        overlap_start = max(seg["start"], segment_start)
        overlap_end = min(seg["end"], segment_end)
        overlap = max(0, overlap_end - overlap_start)
        
        if overlap > max_overlap:
            max_overlap = overlap
            best_speaker = seg["speaker"]
    
    return best_speaker
```

### 2. Speaker Clustering and Consistency

Implement a post-processing step to merge speaker IDs that likely belong to the same person:

```python
def merge_similar_speakers(diarization_result, similarity_threshold=0.85):
    embeddings = {}  # Speaker ID to voice embedding
    
    # Extract voice embeddings for each speaker segment
    for turn, embedding, speaker in diarization_result.itertracks(yield_embedding=True, yield_label=True):
        if speaker not in embeddings:
            embeddings[speaker] = []
        embeddings[speaker].append(embedding)
    
    # Average embeddings per speaker
    for speaker in embeddings:
        embeddings[speaker] = np.mean(embeddings[speaker], axis=0)
    
    # Compute similarity matrix
    speakers = list(embeddings.keys())
    n_speakers = len(speakers)
    similarity_matrix = np.zeros((n_speakers, n_speakers))
    
    for i in range(n_speakers):
        for j in range(i+1, n_speakers):
            similarity = cosine_similarity(
                embeddings[speakers[i]].reshape(1, -1),
                embeddings[speakers[j]].reshape(1, -1)
            )[0][0]
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity
    
    # Cluster speakers
    clusters = {}
    for i, speaker in enumerate(speakers):
        assigned = False
        for cluster_id, cluster_members in clusters.items():
            # Check similarity with all members of the cluster
            similarities = [similarity_matrix[i, speakers.index(member)] for member in cluster_members]
            if all(sim > similarity_threshold for sim in similarities):
                clusters[cluster_id].append(speaker)
                assigned = True
                break
        
        if not assigned:
            # Create new cluster
            clusters[speaker] = [speaker]
    
    # Create mapping of original speakers to final speakers
    speaker_mapping = {}
    for cluster_id, members in clusters.items():
        for member in members:
            speaker_mapping[member] = cluster_id
    
    return speaker_mapping
```

### 3. Fixed Number of Speakers Support

When the number of speakers is known in advance:

```python
def optimize_for_num_speakers(diarization_result, target_num_speakers):
    # Get current number of unique speakers
    current_speakers = set(s for _, _, s in diarization_result.itertracks(yield_label=True))
    current_num = len(current_speakers)
    
    if current_num == target_num_speakers:
        return diarization_result
    
    # Get speaker embeddings
    embeddings = {}
    for turn, embedding, speaker in diarization_result.itertracks(yield_embedding=True, yield_label=True):
        if speaker not in embeddings:
            embeddings[speaker] = []
        embeddings[speaker].append(embedding)
    
    # Average embeddings
    for speaker in embeddings:
        embeddings[speaker] = np.mean(embeddings[speaker], axis=0)
    
    if current_num > target_num_speakers:
        # Need to merge speakers
        # Use hierarchical clustering to merge until we have target_num_speakers
        return cluster_speakers_hierarchical(diarization_result, embeddings, target_num_speakers)
    else:
        # Less speakers than expected, could be error or speakers weren't detected
        # For now, return as is but log a warning
        logger.warning(f"Found {current_num} speakers but expected {target_num_speakers}")
        return diarization_result
```

### 4. Audio Preprocessing for Improved Diarization

Add preprocessing steps to improve audio quality before diarization:

```python
def preprocess_audio_for_diarization(audio_path, output_path=None):
    """
    Preprocess audio to improve diarization quality
    
    Steps:
    1. Noise reduction
    2. Volume normalization
    3. High-pass filter to remove low frequency noise
    4. Channel selection/mixing for multi-channel audio
    """
    if output_path is None:
        output_path = audio_path.with_suffix('.processed.wav')
    
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)
    
    # Noise reduction
    y_reduced = librosa.effects.trim(y, top_db=20)[0]
    
    # Volume normalization
    y_normalized = librosa.util.normalize(y_reduced)
    
    # High-pass filter (remove frequencies below 80Hz)
    b, a = scipy.signal.butter(4, 80/(sr/2), 'highpass')
    y_filtered = scipy.signal.filtfilt(b, a, y_normalized)
    
    # Save processed audio
    sf.write(output_path, y_filtered, sr)
    
    return output_path
```

### 5. Enhanced Pyannote Pipeline Configuration

Configure the diarization pipeline with optimal parameters:

```python
def configure_diarization_pipeline(num_speakers=None):
    """Configure diarization pipeline with optimal parameters"""
    # Base configuration
    pipeline_config = {
        "segmentation": {"min_duration_off": 0.5},  # Min silence between speakers
        "clustering": {
            "method": "clustering.AgglomerativeClustering",
            "metric": "cosine",
            "min_cluster_size": 15,  # Min duration (frames) for a speaker
        }
    }
    
    # If we know the number of speakers
    if num_speakers is not None:
        pipeline_config["clustering"]["num_clusters"] = num_speakers
    
    # Get the pipeline with our custom config
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=HUGGINGFACE_TOKEN,
    )
    
    # Set custom parameters
    pipeline.instantiate(pipeline_config)
    
    return pipeline
```

## Implementation Approach

The implementation will follow these steps:

1. **Modularize**: Extract diarization into a dedicated module
2. **Enhance**: Implement the improved algorithms
3. **Configure**: Make parameters configurable via CLI
4. **Test**: Evaluate each improvement against a test dataset
5. **Combine**: Integrate improvements that show measurable benefit

### Priority Order:

1. Enhanced Speaker Association (highest impact)
2. Audio Preprocessing (good baseline improvement)
3. Speaker Clustering and Consistency 
4. Fixed Number of Speakers Support
5. Pyannote Pipeline Configuration (requires most fine-tuning)

## Accuracy Measurement

To evaluate improvements, implement a Word Error Rate (WER) and Diarization Error Rate (DER) calculation:

```python
def calculate_der(reference, hypothesis):
    """
    Calculate Diarization Error Rate
    
    DER = (false_alarm + missed_detection + speaker_error) / total_reference_speech
    """
    # Implementation logic for DER calculation
    # ...
    
    return der_score
```

## Future Directions

For the longer-term vision of learning speaker names:

1. Create a speaker enrollment process to capture voice profiles
2. Build a database of known speakers with their voice characteristics
3. Implement voice profile matching against enrolled speakers
4. Add a mechanism to update speaker profiles over time

These future enhancements would build on the improved diarization capabilities implemented in the current phase. 