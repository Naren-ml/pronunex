"""
Audio Utilities for Alignment.

Common utilities used across alignment modules.
"""

import logging
from typing import Tuple, List, Dict
from dataclasses import dataclass
import torch
import torchaudio

logger = logging.getLogger(__name__)


@dataclass
class AlignedToken:
    """Represents an aligned token with timestamps."""
    token: str
    start: float
    end: float
    score: float = 1.0


def load_audio(
    audio_path: str, 
    target_sample_rate: int = 16000
) -> Tuple[torch.Tensor, int]:
    """
    Load and preprocess audio file for alignment.
    
    Args:
        audio_path: Path to audio file
        target_sample_rate: Target sample rate (16kHz for wav2vec2)
    
    Returns:
        Tuple of (waveform tensor, sample rate)
    """
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Resample if needed
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate,
            new_freq=target_sample_rate
        )
        waveform = resampler(waveform)
        sample_rate = target_sample_rate
    
    return waveform, sample_rate


def strip_stress(phoneme: str) -> str:
    """Remove stress markers from ARPAbet phoneme (e.g., 'AH0' -> 'AH')."""
    return ''.join(c for c in phoneme if not c.isdigit())


def get_phoneme_duration_weights() -> Dict[str, float]:
    """
    Get relative duration weights for phonemes.
    
    Vowels and diphthongs are typically longer than consonants.
    Stops (P, B, T, D, K, G) are typically shortest.
    """
    return {
        # Short consonants (stops, affricates)
        'P': 0.4, 'B': 0.4, 'T': 0.4, 'D': 0.4, 'K': 0.4, 'G': 0.4,
        'CH': 0.5, 'JH': 0.5,
        
        # Fricatives
        'F': 0.6, 'V': 0.6, 'TH': 0.7, 'DH': 0.7,
        'S': 0.6, 'Z': 0.6, 'SH': 0.7, 'ZH': 0.7,
        'HH': 0.5,
        
        # Nasals and liquids
        'M': 0.7, 'N': 0.7, 'NG': 0.7,
        'L': 0.8, 'R': 0.8,
        
        # Glides
        'W': 0.6, 'Y': 0.6,
        
        # Short vowels
        'IH': 0.8, 'EH': 0.8, 'AE': 0.9, 'AH': 0.8, 'UH': 0.8,
        
        # Long vowels
        'IY': 1.0, 'EY': 1.0, 'AA': 1.0, 'AO': 1.0, 'OW': 1.0, 'UW': 1.0,
        'ER': 1.0,
        
        # Diphthongs (longest)
        'AY': 1.2, 'AW': 1.2, 'OY': 1.2,
    }


def fix_overlapping_timestamps(timestamps: List[dict]) -> List[dict]:
    """
    Fix any overlapping or out-of-order timestamps.
    
    Ensures monotonically increasing sequence.
    """
    if not timestamps:
        return timestamps
    
    fixed = []
    prev_end = 0.0
    
    for ts in timestamps:
        start = max(ts['start'], prev_end)
        end = max(ts['end'], start + 0.01)  # Minimum 10ms duration
        
        fixed.append({
            **ts,
            'start': round(start, 3),
            'end': round(end, 3)
        })
        prev_end = end
    
    return fixed
