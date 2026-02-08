"""
Scorer Module for Pronunex.

Calculates pronunciation scores using cosine similarity between embeddings.
All scoring is deterministic - NO LLM involvement.
"""

import logging
from typing import List
import numpy as np
from scipy.spatial.distance import cosine
from django.conf import settings

logger = logging.getLogger(__name__)


def calculate_phoneme_scores(
    user_embeddings: List[np.ndarray],
    reference_embeddings: List[np.ndarray],
    phonemes: List[str],
    timestamps: List[dict] = None
) -> List[dict]:
    """
    Calculate cosine similarity scores for each phoneme.
    
    UPDATED: Added validation and removed fake fallback.
    Now returns 'unscorable' result instead of fake scores when comparison fails.
    
    Args:
        user_embeddings: User's pronunciation embeddings
        reference_embeddings: Reference (gold standard) embeddings
        phonemes: Expected phoneme sequence
        timestamps: Optional timing information
    
    Returns:
        List of score dicts, OR dict with 'status': 'unscorable' if validation fails
    """
    config = settings.SCORING_CONFIG
    threshold = config.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    # VALIDATION 1: Check embedding count match
    if len(user_embeddings) != len(reference_embeddings):
        logger.error(
            f"Embedding count mismatch: user={len(user_embeddings)}, "
            f"ref={len(reference_embeddings)}, phonemes={len(phonemes)}"
        )
        return generate_unscorable_result(phonemes, "embedding_count_mismatch")
    
    if len(user_embeddings) != len(phonemes):
        logger.warning(
            f"Phoneme count mismatch: embeddings={len(user_embeddings)}, "
            f"phonemes={len(phonemes)}. Using min length."
        )
    
    # VALIDATION 2: Check for zero/corrupt embeddings
    zero_count = sum(1 for emb in user_embeddings if np.all(emb == 0))
    if zero_count > len(user_embeddings) * 0.5:  # >50% zero is bad
        logger.error(f"Too many zero embeddings: {zero_count}/{len(user_embeddings)}")
        return generate_unscorable_result(phonemes, "embedding_quality_poor")
    
    # Check for NaN embeddings
    nan_count = sum(1 for emb in user_embeddings if np.any(np.isnan(emb)))
    if nan_count > 0:
        logger.error(f"Found {nan_count} NaN embeddings")
        return generate_unscorable_result(phonemes, "nan_embeddings")
    
    scores = []
    min_len = min(len(user_embeddings), len(reference_embeddings), len(phonemes))
    
    for i in range(min_len):
        user_emb = user_embeddings[i]
        ref_emb = reference_embeddings[i]
        phoneme = phonemes[i]
        
        # Calculate cosine similarity
        try:
            similarity = calculate_cosine_similarity(user_emb, ref_emb)
        except Exception as e:
            logger.warning(f"Similarity failed for phoneme {i}: {str(e)}")
            similarity = 0.0
        
        score_entry = {
            'phoneme': phoneme,
            'score': round(float(similarity), 3),
            'is_weak': bool(similarity < threshold),
            'index': i,  # Add index for mistake detection
        }
        
        # Add timing info if available
        if timestamps and i < len(timestamps):
            ts = timestamps[i]
            score_entry['start'] = ts.get('start')
            score_entry['end'] = ts.get('end')
            score_entry['word'] = ts.get('word', '')
            score_entry['position'] = ts.get('position', 'medial')
        
        scores.append(score_entry)
    
    # HONEST FAILURE: If all scores are 0, return unscorable (no fake scores!)
    if scores and all(s['score'] == 0 for s in scores):
        logger.error("All phoneme scores are 0 - real failure, not using fake scores")
        return generate_unscorable_result(phonemes, "all_scores_zero")
    
    logger.debug(f"Calculated scores for {len(scores)} phonemes (validated)")
    
    return scores


def generate_unscorable_result(phonemes: List[str], reason: str) -> dict:
    """
    Generate honest 'unscorable' result instead of fake scores.
    
    This replaces the old generate_adaptive_scores function.
    When embedding comparison fails, we TELL THE USER instead of faking scores.
    
    Args:
        phonemes: Expected phoneme sequence
        reason: Why scoring failed
    
    Returns:
        dict with status='unscorable' and helpful message
    """
    messages = {
        'embedding_count_mismatch': 'Audio alignment produced different number of segments than expected.',
        'embedding_quality_poor': 'Could not extract clear speech features from the audio.',
        'nan_embeddings': 'Audio processing produced invalid results.',
        'all_scores_zero': 'Could not compare pronunciation to reference.',
    }
    
    suggestions = {
        'embedding_count_mismatch': 'Try speaking the complete sentence more evenly.',
        'embedding_quality_poor': 'Speak closer to the microphone in a quiet environment.',
        'nan_embeddings': 'Try recording again with stable audio.',
        'all_scores_zero': 'Speak more clearly and at a moderate pace.',
    }
    
    return {
        'status': 'unscorable',
        'reason': reason,
        'phonemes': phonemes,
        'scores': None,
        'message': messages.get(reason, 'Could not accurately score this attempt.'),
        'suggestion': suggestions.get(reason, 'Please speak more clearly and try again.')
    }


# Legacy function - kept for backward compatibility but deprecated
def generate_adaptive_scores(phonemes: List[str], timestamps: List[dict] = None) -> dict:
    """
    DEPRECATED: This function generated FAKE scores which hid real failures.
    
    Use generate_unscorable_result() instead to be honest about failures.
    
    This is now a wrapper that returns an unscorable result.
    """
    logger.warning(
        "generate_adaptive_scores is DEPRECATED! "
        "Do not use fake scores - return unscorable result instead."
    )
    return generate_unscorable_result(phonemes, "adaptive_fallback_deprecated")


def calculate_cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
    
    Returns:
        float: Similarity score between 0 and 1
    """
    # Validate inputs - check for NaN or Inf
    if vec1 is None or vec2 is None:
        return 0.0
    
    if np.any(np.isnan(vec1)) or np.any(np.isnan(vec2)):
        logger.warning("NaN detected in embedding vectors")
        return 0.0
    
    if np.any(np.isinf(vec1)) or np.any(np.isinf(vec2)):
        logger.warning("Inf detected in embedding vectors")
        return 0.0
    
    # Handle zero vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity = 1 - cosine distance
    try:
        distance = cosine(vec1, vec2)
        
        # Check for NaN result (can happen with numerical issues)
        if np.isnan(distance):
            logger.warning("Cosine distance resulted in NaN")
            return 0.0
        
        similarity = 1.0 - distance
        
        # Clamp to [0, 1] range
        return max(0.0, min(1.0, similarity))
    except Exception as e:
        logger.warning(f"Cosine similarity calculation failed: {e}")
        return 0.0


def calculate_overall_score(phoneme_scores: List[dict], weighted: bool = False) -> float:
    """
    Calculate overall pronunciation score from phoneme scores.
    
    Args:
        phoneme_scores: List of per-phoneme scores
        weighted: If True, weight by phoneme importance (future)
    
    Returns:
        float: Overall score between 0 and 1
    """
    if not phoneme_scores:
        return 0.0
    
    if weighted:
        # Future: implement importance weighting
        pass
    
    total = sum(ps['score'] for ps in phoneme_scores)
    return round(total / len(phoneme_scores), 2)


def identify_weak_phonemes(phoneme_scores: List[dict], threshold: float = None) -> List[str]:
    """
    Identify phonemes below the weakness threshold.
    
    Args:
        phoneme_scores: List of per-phoneme scores
        threshold: Custom threshold (uses config if None)
    
    Returns:
        List of weak phoneme symbols
    """
    if threshold is None:
        threshold = settings.SCORING_CONFIG.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    return [ps['phoneme'] for ps in phoneme_scores if ps['score'] < threshold]


def aggregate_phoneme_stats(phoneme_scores: List[dict]) -> dict:
    """
    Aggregate statistics for phoneme scoring.
    
    Args:
        phoneme_scores: List of per-phoneme scores
    
    Returns:
        dict: Statistics including min, max, mean, median, weak count
    """
    if not phoneme_scores:
        return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'weak_count': 0}
    
    scores = [ps['score'] for ps in phoneme_scores]
    weak = identify_weak_phonemes(phoneme_scores)
    
    return {
        'min': round(min(scores), 3),
        'max': round(max(scores), 3),
        'mean': round(np.mean(scores), 3),
        'median': round(np.median(scores), 3),
        'std': round(np.std(scores), 3),
        'weak_count': len(weak),
        'total_count': len(scores),
        'weak_percentage': round(len(weak) / len(scores) * 100, 1),
    }


def compare_with_history(
    current_scores: List[dict], 
    historical_scores: List[List[dict]]
) -> dict:
    """
    Compare current attempt with historical performance.
    
    Args:
        current_scores: Current attempt scores
        historical_scores: List of previous attempt scores
    
    Returns:
        dict: Improvement metrics
    """
    if not historical_scores:
        return {'improvement': 0, 'trend': 'new', 'rank': 1}
    
    current_avg = np.mean([ps['score'] for ps in current_scores])
    historical_avgs = [
        np.mean([ps['score'] for ps in hist]) 
        for hist in historical_scores
    ]
    
    # Compare with most recent
    recent_avg = historical_avgs[-1] if historical_avgs else 0
    improvement = current_avg - recent_avg
    
    # Determine trend
    if len(historical_avgs) >= 3:
        recent_trend = historical_avgs[-3:]
        if all(recent_trend[i] < recent_trend[i+1] for i in range(len(recent_trend)-1)):
            trend = 'improving'
        elif all(recent_trend[i] > recent_trend[i+1] for i in range(len(recent_trend)-1)):
            trend = 'declining'
        else:
            trend = 'stable'
    else:
        trend = 'insufficient_data'
    
    # Calculate rank
    all_avgs = historical_avgs + [current_avg]
    sorted_avgs = sorted(all_avgs, reverse=True)
    rank = sorted_avgs.index(current_avg) + 1
    
    return {
        'improvement': round(improvement, 3),
        'trend': trend,
        'rank': rank,
        'best_score': round(max(all_avgs), 3),
        'average_score': round(np.mean(all_avgs), 3),
    }


# =============================================================================
# WORD-LEVEL SCORING
# =============================================================================

def calculate_word_scores(phoneme_scores: List[dict]) -> List[dict]:
    """
    Combine phoneme results to calculate word-level scores.
    
    Groups phonemes by their word context and computes aggregate
    scores for each word.
    
    Args:
        phoneme_scores: List of per-phoneme scores with word context
    
    Returns:
        List of word score dicts:
        [{"word": "SHE", "score": 0.85, "phonemes": [...], "is_weak": False}, ...]
    """
    config = settings.SCORING_CONFIG
    threshold = config.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    # Group phonemes by word
    word_groups = {}
    
    for ps in phoneme_scores:
        word = ps.get('word', 'unknown')
        if word not in word_groups:
            word_groups[word] = {
                'word': word,
                'phonemes': [],
                'scores': [],
                'start': ps.get('start', 0),
                'end': ps.get('end', 0)
            }
        
        word_groups[word]['phonemes'].append(ps)
        word_groups[word]['scores'].append(ps['score'])
        word_groups[word]['end'] = ps.get('end', word_groups[word]['end'])
    
    # Calculate word-level scores
    word_scores = []
    
    for word, data in word_groups.items():
        if not data['scores']:
            continue
        
        avg_score = np.mean(data['scores'])
        min_score = min(data['scores'])
        weak_phonemes = [
            p['phoneme'] for p in data['phonemes'] 
            if p['score'] < threshold
        ]
        
        word_scores.append({
            'word': word,
            'score': round(float(avg_score), 3),
            'min_score': round(float(min_score), 3),
            'is_weak': bool(avg_score < threshold),
            'weak_phonemes': weak_phonemes,
            'phoneme_count': len(data['phonemes']),
            'start': data['start'],
            'end': data['end']
        })
    
    logger.debug(f"Calculated word scores for {len(word_scores)} words")
    return word_scores


# =============================================================================
# SUBSTITUTION PATTERN DETECTION
# =============================================================================

# Common substitution patterns in speech disorders
SUBSTITUTION_PATTERNS = {
    # Phoneme pairs: (expected, spoken_as)
    ('TH', 'T'): 'Voiceless TH fronting',
    ('TH', 'F'): 'TH/F substitution (lisping)',
    ('DH', 'D'): 'Voiced TH fronting',
    ('DH', 'V'): 'DH/V substitution',
    ('R', 'W'): 'R/W gliding',
    ('L', 'W'): 'L/W gliding',
    ('L', 'Y'): 'L/Y gliding',
    ('S', 'TH'): 'Interdental lisp',
    ('Z', 'DH'): 'Interdental lisp (voiced)',
    ('SH', 'S'): 'Depalatalization',
    ('CH', 'T'): 'Stopping',
    ('JH', 'D'): 'Stopping',
    ('K', 'T'): 'Fronting',
    ('G', 'D'): 'Fronting',
    ('NG', 'N'): 'Velar fronting',
    ('F', 'P'): 'Stopping',
    ('V', 'B'): 'Stopping',
    ('Z', 'S'): 'Devoicing (final consonant)',
    ('B', 'P'): 'Devoicing',
    ('D', 'T'): 'Devoicing',
    ('G', 'K'): 'Devoicing',
}


def detect_substitution_patterns(
    phoneme_scores: List[dict],
    weak_threshold: float = None
) -> List[dict]:
    """
    Detect substitution patterns in weak phonemes.
    
    Analyzes low-scoring phonemes to identify common substitution
    patterns based on difficulty and error type.
    
    Args:
        phoneme_scores: List of per-phoneme scores
        weak_threshold: Score below which phoneme is considered weak
    
    Returns:
        List of detected substitution patterns:
        [{"expected": "TH", "likely_as": "T", "pattern": "Fronting", ...}, ...]
    """
    if weak_threshold is None:
        weak_threshold = settings.SCORING_CONFIG.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    detected = []
    
    for ps in phoneme_scores:
        if ps['score'] >= weak_threshold:
            continue
        
        expected = _strip_stress(ps['phoneme'])
        
        # Check for known substitution patterns
        for (exp, sub), pattern_name in SUBSTITUTION_PATTERNS.items():
            if expected == exp:
                # Calculate likelihood based on score
                # Lower score = more likely substitution
                likelihood = round(1.0 - ps['score'], 2)
                
                detected.append({
                    'expected': expected,
                    'likely_as': sub,
                    'pattern_name': pattern_name,
                    'likelihood': likelihood,
                    'word': ps.get('word', ''),
                    'position': ps.get('position', 'medial'),
                    'score': ps['score']
                })
                break
    
    # Sort by likelihood (most likely first)
    detected.sort(key=lambda x: x['likelihood'], reverse=True)
    
    logger.debug(f"Detected {len(detected)} potential substitution patterns")
    return detected


def _strip_stress(phoneme: str) -> str:
    """Remove stress markers from phoneme."""
    return ''.join(c for c in phoneme if not c.isdigit())


# =============================================================================
# ERROR SUMMARY GENERATION
# =============================================================================

def generate_error_summary(phoneme_scores: List[dict]) -> dict:
    """
    Generate a structured summary of pronunciation errors.
    
    Combines phoneme analysis, word-level analysis, and pattern
    detection into a comprehensive error report.
    
    Args:
        phoneme_scores: List of per-phoneme scores
    
    Returns:
        dict: Comprehensive error summary
    """
    config = settings.SCORING_CONFIG
    threshold = config.get('WEAK_PHONEME_THRESHOLD', 0.7)
    
    # Get basic stats
    stats = aggregate_phoneme_stats(phoneme_scores)
    
    # Get word-level analysis
    word_scores = calculate_word_scores(phoneme_scores)
    weak_words = [w for w in word_scores if w['is_weak']]
    
    # Detect substitution patterns
    substitutions = detect_substitution_patterns(phoneme_scores, threshold)
    
    # Identify missing sounds (scores very low, < 0.3)
    missing_sounds = [
        {
            'phoneme': ps['phoneme'],
            'word': ps.get('word', ''),
            'score': ps['score']
        }
        for ps in phoneme_scores 
        if ps['score'] < 0.3
    ]
    
    # Identify timing issues (based on position patterns)
    timing_issues = _detect_timing_issues(phoneme_scores)
    
    # Group weak phonemes by type
    weak_by_type = _group_weak_by_type(phoneme_scores, threshold)
    
    return {
        'overall_stats': stats,
        'word_level': {
            'total_words': len(word_scores),
            'weak_words_count': len(weak_words),
            'weak_words': weak_words[:5],  # Top 5
        },
        'phoneme_level': {
            'total_phonemes': len(phoneme_scores),
            'weak_count': stats['weak_count'],
            'weak_percentage': stats['weak_percentage'],
            'weak_by_type': weak_by_type,
        },
        'patterns': {
            'substitutions': substitutions[:5],  # Top 5
            'missing_sounds': missing_sounds,
            'timing_issues': timing_issues,
        },
        'recommendations': _generate_recommendations(
            weak_by_type, substitutions, missing_sounds
        )
    }


def _detect_timing_issues(phoneme_scores: List[dict]) -> List[dict]:
    """
    Detect potential timing issues from phoneme patterns.
    
    Looks for patterns like:
    - Final consonant weakness (common in some accents)
    - Initial sound difficulty
    - Cluster reduction patterns
    """
    issues = []
    
    # Group by position
    by_position = {'initial': [], 'medial': [], 'final': []}
    for ps in phoneme_scores:
        pos = ps.get('position', 'medial')
        by_position[pos].append(ps['score'])
    
    # Check for position-based weaknesses
    for position, scores in by_position.items():
        if not scores:
            continue
        
        avg = np.mean(scores)
        if avg < 0.6:
            issues.append({
                'type': f'{position}_weakness',
                'description': f'Weakness in {position} position sounds',
                'average_score': round(avg, 2),
                'count': len(scores)
            })
    
    return issues


def _group_weak_by_type(
    phoneme_scores: List[dict], 
    threshold: float
) -> dict:
    """Group weak phonemes by phoneme type."""
    # Phoneme type mapping
    PHONEME_TYPES = {
        'vowel': {'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 
                  'IH', 'IY', 'OW', 'OY', 'UH', 'UW'},
        'fricative': {'F', 'V', 'TH', 'DH', 'S', 'Z', 'SH', 'ZH', 'HH'},
        'stop': {'P', 'B', 'T', 'D', 'K', 'G'},
        'nasal': {'M', 'N', 'NG'},
        'liquid': {'L', 'R'},
        'glide': {'W', 'Y'},
        'affricate': {'CH', 'JH'},
    }
    
    # Reverse mapping
    phoneme_to_type = {}
    for ptype, phonemes in PHONEME_TYPES.items():
        for p in phonemes:
            phoneme_to_type[p] = ptype
    
    # Group weak phonemes
    weak_by_type = {}
    for ps in phoneme_scores:
        if ps['score'] >= threshold:
            continue
        
        base = _strip_stress(ps['phoneme'])
        ptype = phoneme_to_type.get(base, 'other')
        
        if ptype not in weak_by_type:
            weak_by_type[ptype] = []
        
        weak_by_type[ptype].append({
            'phoneme': ps['phoneme'],
            'score': ps['score'],
            'word': ps.get('word', '')
        })
    
    return weak_by_type


def _generate_recommendations(
    weak_by_type: dict,
    substitutions: List[dict],
    missing_sounds: List[dict]
) -> List[str]:
    """Generate practice recommendations based on error analysis."""
    recommendations = []
    
    # Prioritize by error type
    if missing_sounds:
        phonemes = list(set(m['phoneme'] for m in missing_sounds[:3]))
        recommendations.append(
            f"Focus on producing the following sounds: {', '.join(phonemes)}"
        )
    
    if substitutions:
        patterns = list(set(s['pattern_name'] for s in substitutions[:2]))
        recommendations.append(
            f"Work on correcting: {', '.join(patterns)}"
        )
    
    # Type-specific recommendations
    if 'fricative' in weak_by_type:
        recommendations.append(
            "Practice fricative sounds (S, Z, F, V, TH) with sustained airflow"
        )
    
    if 'liquid' in weak_by_type:
        recommendations.append(
            "Focus on R and L sounds using tongue position exercises"
        )
    
    if 'stop' in weak_by_type:
        recommendations.append(
            "Practice stop consonants (P, B, T, D, K, G) with clear release"
        )
    
    if not recommendations:
        recommendations.append("Continue practicing to maintain your progress")
    
    return recommendations[:5]  # Max 5 recommendations

