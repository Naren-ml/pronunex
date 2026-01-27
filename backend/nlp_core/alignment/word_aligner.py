"""
Word-Level Alignment.

Extracts word boundaries from audio using forced alignment.
"""

import logging
from typing import List

from .utils import load_audio, fix_overlapping_timestamps
from .ctc_aligner import perform_ctc_forced_alignment

logger = logging.getLogger(__name__)


def get_word_timestamps(audio_path: str, text: str) -> List[dict]:
    """
    Get word-level timestamps using forced alignment.
    
    Args:
        audio_path: Path to audio file
        text: Text transcript
    
    Returns:
        List of word timestamps:
        [{"word": "SHE", "start": 0.0, "end": 0.35, "confidence": 0.9}, ...]
    """
    # Load audio
    waveform, sample_rate = load_audio(audio_path)
    
    # Get character-level alignment
    char_alignments = perform_ctc_forced_alignment(waveform, text, sample_rate)
    
    if not char_alignments:
        logger.warning("No character alignments found, using fallback")
        return _fallback_word_timestamps(audio_path, text)
    
    # Group characters into words
    words = text.upper().split()
    word_timestamps = []
    char_idx = 0
    
    for word in words:
        word_chars = list(word)
        
        if char_idx >= len(char_alignments):
            break
        
        # Skip spaces/separators
        while char_idx < len(char_alignments):
            if char_alignments[char_idx].token in ['|', ' ', '']:
                char_idx += 1
            else:
                break
        
        if char_idx >= len(char_alignments):
            break
        
        word_start = char_alignments[char_idx].start
        word_end = word_start
        word_score = 0.0
        matched_chars = 0
        
        # Match characters in word
        for expected_char in word_chars:
            if char_idx < len(char_alignments):
                aligned = char_alignments[char_idx]
                word_end = aligned.end
                word_score += aligned.score
                matched_chars += 1
                char_idx += 1
        
        avg_score = word_score / matched_chars if matched_chars > 0 else 0.5
        
        word_timestamps.append({
            'word': word,
            'start': round(word_start, 3),
            'end': round(word_end, 3),
            'confidence': round(avg_score, 3)
        })
    
    # Fix overlapping timestamps
    word_timestamps = fix_overlapping_timestamps(word_timestamps)
    
    logger.debug(f"Aligned {len(word_timestamps)} words from audio")
    return word_timestamps


def _fallback_word_timestamps(audio_path: str, text: str) -> List[dict]:
    """
    Fallback word timestamp estimation when alignment fails.
    """
    import librosa
    
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        duration = len(y) / sr
    except Exception:
        duration = 2.0
    
    words = text.upper().split()
    if not words:
        return []
    
    word_duration = duration / len(words)
    
    return [
        {
            'word': word,
            'start': round(i * word_duration, 3),
            'end': round((i + 1) * word_duration, 3),
            'confidence': 0.3
        }
        for i, word in enumerate(words)
    ]
