"""
Forced Alignment Module for Pronunex.

This module provides backward-compatible imports for the alignment
subpackage. All implementation is in nlp_core/alignment/.

Usage:
    from nlp_core.aligner import get_phoneme_timestamps
    from nlp_core.aligner import get_word_timestamps
    from nlp_core.aligner import get_phoneme_timestamps_with_text
"""

# Re-export all functions from alignment subpackage
from nlp_core.alignment import (
    get_forced_alignment_model,
    perform_ctc_forced_alignment,
    get_word_timestamps,
    get_phoneme_timestamps,
    get_phoneme_timestamps_with_text,
    load_audio,
    strip_stress,
)

# Additional convenience imports
from nlp_core.alignment.phoneme_aligner import align_audio

__all__ = [
    'get_forced_alignment_model',
    'perform_ctc_forced_alignment',
    'get_word_timestamps',
    'get_phoneme_timestamps',
    'get_phoneme_timestamps_with_text',
    'load_audio',
    'strip_stress',
    'align_audio',
]
