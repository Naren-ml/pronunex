"""
Alignment Package for Pronunex.

Provides forced alignment capabilities for pronunciation assessment.

Modules:
- models: Model loading and management
- ctc_aligner: CTC-based forced alignment
- word_aligner: Word-level alignment
- phoneme_aligner: Phoneme-level alignment
- utils: Utility functions
"""

from .models import get_forced_alignment_model
from .ctc_aligner import perform_ctc_forced_alignment
from .word_aligner import get_word_timestamps
from .phoneme_aligner import (
    get_phoneme_timestamps,
    get_phoneme_timestamps_with_text
)
from .utils import load_audio, strip_stress

__all__ = [
    'get_forced_alignment_model',
    'perform_ctc_forced_alignment',
    'get_word_timestamps',
    'get_phoneme_timestamps',
    'get_phoneme_timestamps_with_text',
    'load_audio',
    'strip_stress',
]
