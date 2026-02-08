"""
Vectorizer Module for Pronunex.

REWRITTEN to fix the "Context-Blind" bug.

OLD (BROKEN): Slice audio → Feed slices to model → Garbage embeddings
NEW (FIXED):  Feed full audio → Get full embeddings → Slice tensor

Wav2Vec2 is a Transformer that needs CONTEXT to understand sounds.
When you feed it tiny slices, it hears noise, not speech.
The fix is to process the FULL audio and then slice the embedding TENSOR.
"""

import logging
from typing import List, Dict, Tuple, Optional
import pickle
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor
from django.conf import settings

logger = logging.getLogger(__name__)

# Singleton model for embedding generation
_embedding_processor = None
_embedding_model = None

# Wav2Vec2 stride: ~320 samples at 16kHz = 0.02 seconds per frame
WAV2VEC2_STRIDE_SECONDS = 320 / 16000  # 0.02


def get_embedding_model():
    """Get or load the Wav2Vec2 model for embeddings."""
    global _embedding_processor, _embedding_model
    
    if _embedding_processor is None or _embedding_model is None:
        logger.info("Loading Wav2Vec2 embedding model...")
        _embedding_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _embedding_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        _embedding_model.eval()
        logger.info("Wav2Vec2 embedding model loaded")
    
    return _embedding_processor, _embedding_model


# =============================================================================
# NEW: Contextual Embedding Functions (Tensor Slicing)
# =============================================================================

def compute_contextual_embeddings(audio_path: str) -> Tuple[np.ndarray, float]:
    """
    Compute FULL contextual embeddings for entire audio.
    
    This preserves the context that Wav2Vec2 needs to understand speech.
    The model "hears" the entire audio and produces frame-level embeddings.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Tuple of (embedding_matrix, stride_seconds)
        - embedding_matrix: Shape [num_frames, 768]
        - stride_seconds: Time per frame (~0.02s)
    """
    processor, model = get_embedding_model()
    
    try:
        # Load full audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
            sr = 16000
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Get full embeddings - model processes ENTIRE audio with full context
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=sr,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Shape: [1, num_frames, 768]
            full_embeddings = outputs.last_hidden_state[0].numpy()
        
        logger.debug(f"Full contextual embeddings shape: {full_embeddings.shape}")
        
        return full_embeddings, WAV2VEC2_STRIDE_SECONDS
        
    except Exception as e:
        logger.error(f"Contextual embedding computation failed: {str(e)}")
        return np.zeros((1, 768)), WAV2VEC2_STRIDE_SECONDS


def slice_embeddings_by_timestamps(
    full_embeddings: np.ndarray,
    timestamps: List[Dict],
    stride_seconds: float = WAV2VEC2_STRIDE_SECONDS
) -> List[np.ndarray]:
    """
    Slice the embedding TENSOR using timestamps.
    
    This is the KEY FIX: We slice the TENSOR, not the audio!
    The embeddings already contain contextual information from the full audio.
    
    Args:
        full_embeddings: Shape [num_frames, 768] from compute_contextual_embeddings
        timestamps: List of {'start': float, 'end': float, 'phoneme': str}
        stride_seconds: Time per embedding frame (default 0.02s)
    
    Returns:
        List of 768-dim embedding vectors, one per phoneme
    """
    phoneme_embeddings = []
    num_frames = full_embeddings.shape[0]
    
    for ts in timestamps:
        # Convert time to frame indices
        start_frame = int(ts['start'] / stride_seconds)
        end_frame = int(ts['end'] / stride_seconds)
        
        # Ensure valid range
        start_frame = max(0, start_frame)
        end_frame = min(num_frames, end_frame)
        
        # Ensure at least one frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1
            if end_frame > num_frames:
                start_frame = max(0, num_frames - 1)
                end_frame = num_frames
        
        # Slice the TENSOR (contextual embeddings preserved!)
        frame_slice = full_embeddings[start_frame:end_frame, :]
        
        # Mean pool to get single 768-dim vector
        if frame_slice.shape[0] > 0:
            pooled = np.mean(frame_slice, axis=0)
        else:
            pooled = np.zeros(768)
        
        phoneme_embeddings.append(pooled)
    
    logger.debug(f"Sliced {len(phoneme_embeddings)} phoneme embeddings from tensor")
    
    return phoneme_embeddings


def compute_phoneme_embeddings(
    audio_path: str,
    timestamps: List[Dict]
) -> List[np.ndarray]:
    """
    Main function: Get phoneme embeddings using contextual tensor slicing.
    
    This is the CORRECT way to get phoneme embeddings:
    1. Feed FULL audio to model → preserves context
    2. Get full embedding matrix
    3. Slice the TENSOR by timestamps → each phoneme gets contextual embedding
    
    Args:
        audio_path: Path to cleaned audio file
        timestamps: List of phoneme timestamps
    
    Returns:
        List of 768-dim embeddings, one per phoneme
    """
    # Step 1: Get full contextual embeddings
    full_embeddings, stride = compute_contextual_embeddings(audio_path)
    
    # Step 2: Slice tensor by timestamps
    return slice_embeddings_by_timestamps(full_embeddings, timestamps, stride)


# =============================================================================
# LEGACY: Old Functions (Deprecated but kept for backward compatibility)
# =============================================================================

def audio_to_embedding(audio_slice: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """
    DEPRECATED: This function has the context-blind bug.
    
    Use compute_phoneme_embeddings() with tensor slicing instead.
    
    Kept for backward compatibility with reference embedding generation.
    """
    processor, model = get_embedding_model()
    
    try:
        # Ensure minimum length for processing
        if len(audio_slice) < 400:
            audio_slice = np.pad(audio_slice, (0, 400 - len(audio_slice)))
        
        # Process audio
        inputs = processor(
            audio_slice.astype(np.float32), 
            sampling_rate=sample_rate, 
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.last_hidden_state
            embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
        
        return embedding
        
    except Exception as e:
        logger.error(f"Embedding generation failed: {str(e)}")
        return np.zeros(settings.SCORING_CONFIG.get('EMBEDDING_DIM', 768))


def batch_audio_to_embeddings(audio_slices: List[np.ndarray]) -> List[np.ndarray]:
    """
    DEPRECATED: Use compute_phoneme_embeddings() instead.
    
    This function slices audio which destroys Wav2Vec2 context.
    Kept for backward compatibility.
    """
    logger.warning(
        "batch_audio_to_embeddings is DEPRECATED! "
        "Use compute_phoneme_embeddings() for contextual embeddings."
    )
    
    embeddings = []
    for i, audio_slice in enumerate(audio_slices):
        try:
            embedding = audio_to_embedding(audio_slice)
            embeddings.append(embedding)
        except Exception as e:
            logger.warning(f"Failed to embed slice {i}: {str(e)}")
            embeddings.append(np.zeros(settings.SCORING_CONFIG.get('EMBEDDING_DIM', 768)))
    
    return embeddings


def compute_sentence_embedding(audio_path: str, sample_rate: int = 16000) -> np.ndarray:
    """
    Compute a single embedding for an entire audio file.
    
    This is used when phoneme-level alignment is not available.
    Now uses the full contextual embedding approach.
    
    Args:
        audio_path: Path to audio file
        sample_rate: Target sample rate (default 16kHz)
    
    Returns:
        np.ndarray: 768-dimensional embedding vector
    """
    try:
        # Use contextual embeddings and mean pool the entire thing
        full_embeddings, _ = compute_contextual_embeddings(audio_path)
        return np.mean(full_embeddings, axis=0)
    except Exception as e:
        logger.error(f"Sentence embedding failed: {str(e)}")
        return np.zeros(settings.SCORING_CONFIG.get('EMBEDDING_DIM', 768))


def compute_reference_embeddings(sentence) -> List[np.ndarray]:
    """
    Compute reference embeddings for a sentence.
    
    UPDATED to use contextual tensor slicing.
    
    Args:
        sentence: ReferenceSentence model instance
    
    Returns:
        List of embedding vectors for each phoneme
    """
    from .audio_cleaner import clean_audio
    
    audio_source = sentence.get_audio_source()
    if not audio_source:
        raise ValueError("Sentence has no audio source")
    
    # Clean the reference audio
    cleaned_path = clean_audio(audio_source)
    
    # Get timestamps from alignment map
    timestamps = sentence.alignment_map
    
    # Use NEW contextual method
    embeddings = compute_phoneme_embeddings(cleaned_path, timestamps)
    
    return embeddings


# =============================================================================
# Serialization Functions
# =============================================================================

def serialize_embeddings(embeddings: List[np.ndarray]) -> bytes:
    """
    Serialize embeddings for database storage.
    
    Args:
        embeddings: List of numpy arrays
    
    Returns:
        bytes: Pickled embeddings
    """
    return pickle.dumps(embeddings)


def deserialize_embeddings(data: bytes) -> List[np.ndarray]:
    """
    Deserialize embeddings from database.
    
    Args:
        data: Pickled embeddings
    
    Returns:
        List of numpy arrays
    """
    return pickle.loads(data)


def embedding_distance(emb1: np.ndarray, emb2: np.ndarray, metric: str = 'cosine') -> float:
    """
    Calculate distance between two embeddings.
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        metric: Distance metric ('cosine' or 'euclidean')
    
    Returns:
        float: Distance value
    """
    if metric == 'cosine':
        from scipy.spatial.distance import cosine
        return cosine(emb1, emb2)
    elif metric == 'euclidean':
        return np.linalg.norm(emb1 - emb2)
    else:
        raise ValueError(f"Unknown metric: {metric}")
