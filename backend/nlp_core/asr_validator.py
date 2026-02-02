"""
ASR Validator Module for Pronunex.

Gatekeeper that verifies user's speech matches expected text BEFORE scoring.
Prevents the "Yes Man" bug where wrong speech gets valid scores.

Uses the same Wav2Vec2 model for ASR transcription to verify content.
"""

import logging
from typing import List, Dict, Tuple
from difflib import SequenceMatcher
import torch
import torchaudio

logger = logging.getLogger(__name__)

# Singleton instances
_asr_processor = None
_asr_model = None


def _get_asr_model():
    """Load Wav2Vec2 ASR model (cached singleton)."""
    global _asr_processor, _asr_model
    
    if _asr_model is None:
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        logger.info("Loading Wav2Vec2 ASR model for validation...")
        _asr_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        _asr_model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        _asr_model.eval()
        logger.info("Wav2Vec2 ASR model loaded")
    
    return _asr_processor, _asr_model


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file to text using ASR.
    
    Args:
        audio_path: Path to audio file
    
    Returns:
        Transcribed text (lowercase, cleaned)
    """
    processor, model = _get_asr_model()
    
    try:
        # Load audio
        waveform, sr = torchaudio.load(audio_path)
        
        # Resample to 16kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Process through model
        inputs = processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            logits = model(**inputs).logits
            pred_ids = torch.argmax(logits, dim=-1)
        
        # Decode to text
        transcription = processor.decode(pred_ids[0])
        
        logger.debug(f"ASR transcription: '{transcription}'")
        return transcription.lower().strip()
        
    except Exception as e:
        logger.error(f"ASR transcription failed: {str(e)}")
        return ""


def get_word_diff(transcribed_words: List[str], expected_words: List[str]) -> List[Dict]:
    """
    Get word-level differences for mistake detection.
    
    Example:
        Expected:    ["she", "sells", "books"]
        Transcribed: ["she", "sell", "book"]
        
        Returns differences showing missing 's' in sells and books.
    
    Args:
        transcribed_words: Words from ASR
        expected_words: Expected words from sentence
    
    Returns:
        List of word diff dicts with type, position, and details
    """
    results = []
    
    matcher = SequenceMatcher(None, transcribed_words, expected_words)
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            # Words match exactly
            for idx, word in enumerate(expected_words[j1:j2]):
                results.append({
                    'word': word,
                    'type': 'correct',
                    'position': j1 + idx
                })
        
        elif tag == 'replace':
            # Words differ
            for idx in range(max(i2 - i1, j2 - j1)):
                user_word = transcribed_words[i1 + idx] if i1 + idx < i2 else None
                expected_word = expected_words[j1 + idx] if j1 + idx < j2 else None
                
                if expected_word:
                    issue = _detect_word_issue(user_word, expected_word)
                    results.append({
                        'word': expected_word,
                        'type': 'wrong',
                        'position': j1 + idx,
                        'user_said': user_word or '(nothing)',
                        'suggestion': expected_word,
                        'issue': issue
                    })
        
        elif tag == 'delete':
            # User said extra words
            for idx, word in enumerate(transcribed_words[i1:i2]):
                results.append({
                    'word': word,
                    'type': 'extra',
                    'position': i1 + idx,
                    'user_said': word
                })
        
        elif tag == 'insert':
            # User missed words
            for idx, word in enumerate(expected_words[j1:j2]):
                results.append({
                    'word': word,
                    'type': 'missing',
                    'position': j1 + idx,
                    'suggestion': f"You missed saying '{word}'"
                })
    
    return sorted(results, key=lambda x: x['position'])


def _detect_word_issue(user_word: str, expected_word: str) -> str:
    """
    Detect specific issue between user's word and expected word.
    
    Examples:
        - "sell" vs "sells" → "missing_ending_s"
        - "ook" vs "book" → "missing_beginning_b"
        - "dat" vs "that" → "substituted_th_with_d"
    """
    if not user_word:
        return 'word_skipped'
    
    user_word = user_word.lower()
    expected_word = expected_word.lower()
    
    # Check for missing ending letters
    if expected_word.startswith(user_word) and len(expected_word) > len(user_word):
        missing = expected_word[len(user_word):]
        return f"missing_ending_{missing}"
    
    # Check for missing beginning letters
    if expected_word.endswith(user_word) and len(expected_word) > len(user_word):
        missing = expected_word[:-len(user_word)]
        return f"missing_beginning_{missing}"
    
    # Check for single character substitutions
    if len(user_word) == len(expected_word):
        diffs = [
            (i, u, e) 
            for i, (u, e) in enumerate(zip(user_word, expected_word)) 
            if u != e
        ]
        if len(diffs) == 1:
            pos, user_char, expected_char = diffs[0]
            return f"substituted_{expected_char}_with_{user_char}"
    
    # Check for TH → D/T substitution (common)
    if 'th' in expected_word and ('d' in user_word or 't' in user_word):
        return 'th_substitution'
    
    # General mispronunciation
    return 'mispronounced'


def validate_speech(
    audio_path: str,
    expected_text: str,
    similarity_threshold: float = 0.6
) -> Dict:
    """
    Validate user's speech against expected text.
    
    This is the GATEKEEPER function that prevents wrong speech from being scored.
    
    Args:
        audio_path: Path to cleaned audio file
        expected_text: Expected sentence text
        similarity_threshold: Minimum similarity to proceed (default 0.6)
    
    Returns:
        {
            'status': 'match' | 'partial' | 'mismatch',
            'transcribed': str,
            'expected': str,
            'similarity': float,
            'word_diff': List[dict],
            'missing_words': List[dict],
            'extra_words': List[dict],
            'wrong_words': List[dict],
            'can_proceed': bool
        }
    """
    # Get ASR transcription
    transcribed = transcribe_audio(audio_path)
    
    if not transcribed:
        logger.warning("ASR returned empty transcription")
        return {
            'status': 'error',
            'transcribed': '',
            'expected': expected_text,
            'similarity': 0.0,
            'word_diff': [],
            'missing_words': [],
            'extra_words': [],
            'wrong_words': [],
            'can_proceed': False,
            'message': 'Could not transcribe audio. Please speak more clearly.'
        }
    
    # Normalize for comparison
    trans_normalized = transcribed.lower().strip()
    expected_normalized = expected_text.lower().strip()
    
    # Calculate overall similarity
    similarity = SequenceMatcher(
        None,
        trans_normalized,
        expected_normalized
    ).ratio()
    
    # Get word-level diff
    trans_words = trans_normalized.split()
    expected_words = expected_normalized.split()
    
    word_diff = get_word_diff(trans_words, expected_words)
    
    # Categorize differences
    missing_words = [w for w in word_diff if w['type'] == 'missing']
    extra_words = [w for w in word_diff if w['type'] == 'extra']
    wrong_words = [w for w in word_diff if w['type'] == 'wrong']
    
    # Determine status
    if similarity >= 0.9:
        status = 'match'
        can_proceed = True
    elif similarity >= similarity_threshold:
        status = 'partial'
        can_proceed = True  # Allow scoring but show mistakes
    else:
        status = 'mismatch'
        can_proceed = False  # Reject - too different
    
    result = {
        'status': status,
        'transcribed': transcribed,
        'expected': expected_text,
        'similarity': round(similarity, 3),
        'word_diff': word_diff,
        'missing_words': missing_words,
        'extra_words': extra_words,
        'wrong_words': wrong_words,
        'can_proceed': can_proceed
    }
    
    # Add helpful message
    if status == 'mismatch':
        result['message'] = f"It sounds like you said something different. Please try saying: '{expected_text}'"
    elif status == 'partial':
        error_count = len(missing_words) + len(wrong_words)
        result['message'] = f"Good attempt! Found {error_count} word(s) to improve."
    else:
        result['message'] = "Great! You said the sentence correctly."
    
    logger.info(f"ASR validation: status={status}, similarity={similarity:.2f}")
    
    return result


# Convenience alias
def validate(audio_path: str, expected_text: str, threshold: float = 0.6) -> Dict:
    """Alias for validate_speech."""
    return validate_speech(audio_path, expected_text, threshold)
