"""
Phoneme-Level Alignment.

Extracts phoneme boundaries using word alignment and G2P interpolation.
"""

import logging
from typing import List, Optional

from .utils import (
    load_audio, 
    strip_stress, 
    get_phoneme_duration_weights
)
from .word_aligner import get_word_timestamps

logger = logging.getLogger(__name__)


def get_phoneme_timestamps(
    audio_path: str, 
    expected_phonemes: List[str]
) -> List[dict]:
    """
    Get phoneme-level timestamps using weighted distribution.
    
    Uses phoneme duration weights to distribute time across phonemes
    based on their natural duration characteristics.
    
    Args:
        audio_path: Path to cleaned audio file
        expected_phonemes: Expected ARPAbet phoneme sequence from G2P
    
    Returns:
        List of phoneme timestamps:
        [{"phoneme": "SH", "start": 0.0, "end": 0.12}, ...]
    """
    if not expected_phonemes:
        return []
    
    # Load audio to get duration
    waveform, sample_rate = load_audio(audio_path)
    audio_duration = waveform.shape[1] / sample_rate
    
    # Get phoneme duration weights
    phoneme_weights = get_phoneme_duration_weights()
    
    # Calculate weighted distribution
    total_weight = sum(
        phoneme_weights.get(strip_stress(p), 1.0) 
        for p in expected_phonemes
    )
    
    phoneme_timestamps = []
    current_time = 0.0
    
    for i, phoneme in enumerate(expected_phonemes):
        base_phoneme = strip_stress(phoneme)
        weight = phoneme_weights.get(base_phoneme, 1.0)
        duration = (weight / total_weight) * audio_duration
        
        phoneme_timestamps.append({
            'phoneme': phoneme,
            'start': round(current_time, 3),
            'end': round(current_time + duration, 3),
            'index': i,
            'weight': weight
        })
        
        current_time += duration
    
    logger.debug(f"Generated timestamps for {len(expected_phonemes)} phonemes")
    return phoneme_timestamps


def get_phoneme_timestamps_with_text(
    audio_path: str,
    text: str,
    expected_phonemes: Optional[List[str]] = None
) -> List[dict]:
    """
    Get phoneme-level timestamps using word-level forced alignment.
    
    This is the most accurate method as it:
    1. Uses real forced alignment for word boundaries
    2. Uses precomputed phonemes OR G2P to get phonemes per word
    3. Distributes phonemes within accurate word boundaries
    
    Args:
        audio_path: Path to audio file
        text: Text transcript
        expected_phonemes: Precomputed phoneme sequence (preferred) or None for G2P
    
    Returns:
        List of phoneme timestamps with word context
    """
    from nlp_core.phoneme_extractor import text_to_phonemes_with_words
    
    # Step 1: Get word-level timestamps
    word_timestamps = get_word_timestamps(audio_path, text)
    
    if not word_timestamps:
        logger.warning("No word timestamps from alignment, using fallback")
        return get_phoneme_timestamps(audio_path, expected_phonemes or [])
    
    # Step 2: Determine phonemes to use
    # PRIORITY: Use expected_phonemes if provided (from database)
    # This ensures consistency between reference and user scoring
    if expected_phonemes:
        # Distribute expected phonemes across words proportionally
        return _distribute_phonemes_across_words(
            expected_phonemes, word_timestamps, text
        )
    
    # Fallback: Use G2P to get phonemes per word
    word_phoneme_map = text_to_phonemes_with_words(text)
    
    # Step 3: Distribute phonemes within word boundaries
    phoneme_timestamps = []
    phoneme_index = 0
    phoneme_weights = get_phoneme_duration_weights()
    
    # Match word timestamps with G2P output
    min_len = min(len(word_timestamps), len(word_phoneme_map))
    
    for i in range(min_len):
        word_ts = word_timestamps[i]
        word, phonemes = word_phoneme_map[i]
        
        word_start = word_ts['start']
        word_end = word_ts['end']
        word_duration = word_end - word_start
        
        if not phonemes:
            continue
        
        # Calculate weighted distribution within word
        total_weight = sum(
            phoneme_weights.get(strip_stress(p), 1.0) 
            for p in phonemes
        )
        
        current_time = word_start
        
        for j, phoneme in enumerate(phonemes):
            base_phoneme = strip_stress(phoneme)
            weight = phoneme_weights.get(base_phoneme, 1.0)
            duration = (weight / total_weight) * word_duration
            
            # Determine position in word
            if j == 0:
                position = 'initial'
            elif j == len(phonemes) - 1:
                position = 'final'
            else:
                position = 'medial'
            
            phoneme_timestamps.append({
                'phoneme': phoneme,
                'start': round(current_time, 3),
                'end': round(current_time + duration, 3),
                'index': phoneme_index,
                'word': word,
                'position': position,
                'confidence': word_ts.get('confidence', 0.5)
            })
            
            current_time += duration
            phoneme_index += 1
    
    logger.debug(f"Aligned {len(phoneme_timestamps)} phonemes using word bounds")
    return phoneme_timestamps


def _distribute_phonemes_across_words(
    phonemes: List[str],
    word_timestamps: List[dict],
    text: str
) -> List[dict]:
    """
    Distribute precomputed phonemes across word boundaries.
    
    This maintains consistency with database-stored phoneme sequences
    while using accurate word-level timing from forced alignment.
    """
    phoneme_weights = get_phoneme_duration_weights()
    
    # Calculate total audio duration from word timestamps
    if not word_timestamps:
        return []
    
    total_start = word_timestamps[0]['start']
    total_end = word_timestamps[-1]['end']
    total_duration = total_end - total_start
    
    if total_duration <= 0 or not phonemes:
        return []
    
    # Calculate weighted duration for each phoneme
    total_weight = sum(
        phoneme_weights.get(strip_stress(p), 1.0) 
        for p in phonemes
    )
    
    # Get word boundaries for position assignment
    words = text.split()
    
    phoneme_timestamps = []
    current_time = total_start
    
    for i, phoneme in enumerate(phonemes):
        base_phoneme = strip_stress(phoneme)
        weight = phoneme_weights.get(base_phoneme, 1.0)
        duration = (weight / total_weight) * total_duration
        
        # Find which word this phoneme belongs to (approximate)
        word_idx = min(i * len(words) // len(phonemes), len(words) - 1)
        word = words[word_idx] if words else ""
        
        # Determine position
        if i == 0:
            position = 'initial'
        elif i == len(phonemes) - 1:
            position = 'final'
        else:
            position = 'medial'
        
        phoneme_timestamps.append({
            'phoneme': phoneme,
            'start': round(current_time, 3),
            'end': round(current_time + duration, 3),
            'index': i,
            'word': word,
            'position': position,
            'confidence': 0.7  # Moderate confidence for distributed phonemes
        })
        
        current_time += duration
    
    logger.debug(f"Distributed {len(phonemes)} precomputed phonemes across {len(word_timestamps)} words")
    return phoneme_timestamps


def align_audio(audio_path: str, text: str) -> dict:
    """
    Main alignment function providing both word and phoneme timestamps.
    
    Args:
        audio_path: Path to audio file
        text: Text transcript
    
    Returns:
        Dict containing word and phoneme level alignments
    """
    word_ts = get_word_timestamps(audio_path, text)
    phoneme_ts = get_phoneme_timestamps_with_text(audio_path, text)
    
    return {
        'word_timestamps': word_ts,
        'phoneme_timestamps': phoneme_ts,
        'audio_path': audio_path,
        'text': text
    }
