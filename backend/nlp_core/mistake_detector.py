"""
Mistake Detector Module for Pronunex.

Pinpoints exactly WHERE user made pronunciation errors:
- Word-level: "You said 'sell' instead of 'sells'"
- Phoneme-level: "The 'TH' sound needs work"
- Character-level: "Missing 's' at the end"

Uses LLM for generating dynamic, personalized pronunciation tips.
"""

import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Mistake:
    """Represents a single pronunciation mistake."""
    type: str  # 'missing_word', 'wrong_word', 'missing_sound', 'wrong_sound', 'weak_phoneme'
    position: int  # Index in sentence
    expected: str  # What should have been said
    actual: str  # What user actually said
    severity: str  # 'minor', 'moderate', 'major'
    suggestion: str  # Help text for user
    phoneme: Optional[str] = None  # For phoneme-level errors
    word: Optional[str] = None  # Associated word
    
    def to_dict(self) -> dict:
        return asdict(self)


class MistakeDetector:
    """
    Detects and explains pronunciation mistakes.
    
    Combines ASR word diff + phoneme scores to give comprehensive feedback.
    Uses LLM for generating personalized pronunciation tips.
    """
    
    def __init__(self, weak_threshold: float = 0.7):
        self.weak_threshold = weak_threshold
        self._llm_service = None
    
    def _get_llm_service(self):
        """Lazy load LLM service."""
        if self._llm_service is None:
            from services.llm_service import LLMService
            self._llm_service = LLMService()
        return self._llm_service
    
    def detect_mistakes(
        self,
        asr_result: Dict,  # From asr_validator
        phoneme_scores: List[Dict],  # From scorer
        expected_text: str
    ) -> Dict:
        """
        Combine ASR word diff + phoneme scores to pinpoint all mistakes.
        
        Args:
            asr_result: Result from asr_validator.validate_speech()
            phoneme_scores: List of phoneme score dicts from scorer
            expected_text: The expected sentence text
        
        Returns:
            Comprehensive mistake report with feedback
        """
        mistakes = []
        
        # 1. Word-level mistakes from ASR comparison
        word_mistakes = self._detect_word_mistakes(asr_result)
        mistakes.extend(word_mistakes)
        
        # 2. Phoneme-level mistakes from scores
        phoneme_mistakes = self._detect_phoneme_mistakes(phoneme_scores)
        mistakes.extend(phoneme_mistakes)
        
        # 3. Generate user-friendly feedback (with AI tips for phonemes)
        feedback = self._generate_feedback(mistakes, expected_text, phoneme_scores)
        
        return {
            'has_mistakes': len(mistakes) > 0,
            'mistakes': [m.to_dict() for m in mistakes],
            'mistake_count': len(mistakes),
            'word_errors': len(word_mistakes),
            'phoneme_errors': len(phoneme_mistakes),
            'feedback': feedback,
            'summary': self._generate_summary(mistakes)
        }
    
    def _detect_word_mistakes(self, asr_result: Dict) -> List[Mistake]:
        """Extract word-level mistakes from ASR comparison."""
        mistakes = []
        
        for diff in asr_result.get('word_diff', []):
            if diff['type'] == 'missing':
                mistakes.append(Mistake(
                    type='missing_word',
                    position=diff['position'],
                    expected=diff['word'],
                    actual='(not spoken)',
                    severity='major',
                    suggestion=f"You skipped the word '{diff['word']}'. Try saying the complete sentence.",
                    word=diff['word']
                ))
            
            elif diff['type'] == 'wrong':
                issue = diff.get('issue', 'mispronounced')
                user_said = diff.get('user_said', '')
                expected_word = diff['word']
                
                # Determine severity and generate specific suggestion
                if 'missing_ending' in issue:
                    missing = issue.replace('missing_ending_', '')
                    mistakes.append(Mistake(
                        type='missing_sound',
                        position=diff['position'],
                        expected=expected_word,
                        actual=user_said,
                        severity='moderate',
                        suggestion=f"You said '{user_said}' but it should be '{expected_word}'. Don't forget the '{missing}' at the end!",
                        word=expected_word
                    ))
                
                elif 'missing_beginning' in issue:
                    missing = issue.replace('missing_beginning_', '')
                    mistakes.append(Mistake(
                        type='missing_sound',
                        position=diff['position'],
                        expected=expected_word,
                        actual=user_said,
                        severity='moderate',
                        suggestion=f"You said '{user_said}' but it should be '{expected_word}'. Start with '{missing}'.",
                        word=expected_word
                    ))
                
                elif 'substituted' in issue:
                    # e.g., "substituted_th_with_d"
                    parts = issue.replace('substituted_', '').split('_with_')
                    if len(parts) == 2:
                        expected_sound, user_sound = parts
                        mistakes.append(Mistake(
                            type='wrong_sound',
                            position=diff['position'],
                            expected=expected_word,
                            actual=user_said,
                            severity='minor',
                            suggestion=f"In '{expected_word}', you used '{user_sound}' instead of '{expected_sound}'.",
                            word=expected_word
                        ))
                    else:
                        mistakes.append(Mistake(
                            type='wrong_word',
                            position=diff['position'],
                            expected=expected_word,
                            actual=user_said,
                            severity='moderate',
                            suggestion=f"'{user_said}' should be '{expected_word}'.",
                            word=expected_word
                        ))
                
                elif issue == 'th_substitution':
                    mistakes.append(Mistake(
                        type='wrong_sound',
                        position=diff['position'],
                        expected=expected_word,
                        actual=user_said,
                        severity='moderate',
                        suggestion=f"In '{expected_word}', the 'TH' sound was pronounced as 'D' or 'T'. Put your tongue between your teeth.",
                        word=expected_word,
                        phoneme='TH'
                    ))
                
                else:
                    mistakes.append(Mistake(
                        type='wrong_word',
                        position=diff['position'],
                        expected=expected_word,
                        actual=user_said,
                        severity='major',
                        suggestion=f"You said '{user_said}' instead of '{expected_word}'. Practice this word.",
                        word=expected_word
                    ))
            
            elif diff['type'] == 'extra':
                mistakes.append(Mistake(
                    type='extra_word',
                    position=diff['position'],
                    expected='(nothing)',
                    actual=diff.get('user_said', ''),
                    severity='minor',
                    suggestion=f"You added an extra word '{diff.get('user_said', '')}'. Try to match the exact sentence.",
                    word=diff.get('user_said', '')
                ))
        
        return mistakes
    
    def _detect_phoneme_mistakes(self, phoneme_scores: List[Dict]) -> List[Mistake]:
        """Extract phoneme-level mistakes from scores."""
        mistakes = []
        
        # Handle unscorable or invalid results
        if not phoneme_scores or isinstance(phoneme_scores, dict):
            return mistakes
        
        for idx, ps in enumerate(phoneme_scores):
            if ps.get('is_weak', False) and ps['score'] < self.weak_threshold:
                # Determine severity based on score
                score = ps['score']
                if score < 0.4:
                    severity = 'major'
                elif score < 0.6:
                    severity = 'moderate'
                else:
                    severity = 'minor'
                
                phoneme = ps['phoneme']
                word = ps.get('word', '')
                
                mistakes.append(Mistake(
                    type='weak_phoneme',
                    position=idx,
                    expected=phoneme,
                    actual=f"(score: {score:.2f})",
                    severity=severity,
                    suggestion='',  # Will be filled by AI
                    phoneme=phoneme,
                    word=word
                ))
        
        return mistakes
    
    def _generate_feedback(
        self, 
        mistakes: List[Mistake], 
        expected_text: str,
        phoneme_scores: List[Dict]
    ) -> Dict:
        """
        Generate user-friendly feedback with AI-powered tips.
        """
        if not mistakes:
            return {
                'status': 'excellent',
                'message': 'Great job! You pronounced everything correctly!',
                'tips': []
            }
        
        # Group by severity
        major = [m for m in mistakes if m.severity == 'major']
        moderate = [m for m in mistakes if m.severity == 'moderate']
        minor = [m for m in mistakes if m.severity == 'minor']
        
        # Determine overall status
        if major:
            status = 'needs_work'
            message = f"Found {len(major)} significant issue(s). Let's work on them!"
        elif moderate:
            status = 'good'
            message = f"Good attempt! Just {len(moderate)} thing(s) to improve."
        else:
            status = 'almost_perfect'
            message = f"Almost perfect! Just {len(minor)} minor detail(s)."
        
        # Generate tips
        tips = []
        
        # Get weak phonemes that need AI tips
        weak_phonemes = [m for m in mistakes if m.type == 'weak_phoneme']
        
        # Generate AI tips for phonemes
        if weak_phonemes:
            ai_tips = self._generate_ai_tips(weak_phonemes, expected_text)
            # Update mistake suggestions with AI tips
            for mistake, tip in zip(weak_phonemes, ai_tips):
                mistake.suggestion = tip
        
        # Collect all tips (top 5)
        all_mistakes = major + moderate + minor
        for m in all_mistakes[:5]:
            tips.append({
                'type': m.type,
                'expected': m.expected,
                'actual': m.actual,
                'word': m.word,
                'phoneme': m.phoneme,
                'severity': m.severity,
                'suggestion': m.suggestion
            })
        
        return {
            'status': status,
            'message': message,
            'tips': tips
        }
    
    def _generate_ai_tips(self, weak_phonemes: List[Mistake], sentence: str) -> List[str]:
        """
        Generate pronunciation tips using LLM.
        
        Uses AI to create personalized, context-aware tips for each weak phoneme.
        """
        if not weak_phonemes:
            return []
        
        try:
            llm = self._get_llm_service()
            
            # Build prompt for AI
            phoneme_list = []
            for m in weak_phonemes:
                phoneme_list.append({
                    'phoneme': m.phoneme,
                    'word': m.word or 'unknown',
                    'score': float(m.actual.replace('(score: ', '').replace(')', '')) if 'score' in m.actual else 0.5
                })
            
            prompt = f"""You are a speech therapy assistant. Generate concise pronunciation tips for these weak phonemes.

Sentence: "{sentence}"

Weak phonemes that need improvement:
{phoneme_list}

For each phoneme, provide ONE short, actionable tip (max 100 characters) that helps the user physically produce the sound correctly.

Focus on:
- Tongue position
- Lip shape
- Airflow
- Common mistakes to avoid

Return a JSON array of tips in the same order as the input phonemes.
Example: ["Put tongue between teeth for TH", "Round lips for SH sound"]

Return ONLY the JSON array, no other text."""

            result = llm.generate(
                prompt=prompt,
                max_tokens=500,
                temperature=0.3,
                response_format="json"
            )
            
            if result.get('success') and result.get('content'):
                import json
                tips = json.loads(result['content'])
                if isinstance(tips, list) and len(tips) == len(weak_phonemes):
                    return tips
            
            # Fallback to basic tips
            return [self._get_fallback_tip(m.phoneme) for m in weak_phonemes]
            
        except Exception as e:
            logger.warning(f"AI tip generation failed: {e}")
            return [self._get_fallback_tip(m.phoneme) for m in weak_phonemes]
    
    def _get_fallback_tip(self, phoneme: str) -> str:
        """Get basic fallback tip if AI fails."""
        # Strip stress markers
        base = ''.join(c for c in phoneme if not c.isdigit())
        
        basic_tips = {
            'TH': "Put your tongue between your teeth and blow air.",
            'DH': "Put tongue between teeth, add voice for 'TH' in 'the'.",
            'R': "Curl your tongue back slightly.",
            'L': "Touch tongue tip to the roof of your mouth.",
            'SH': "Round your lips and push air through.",
            'CH': "Start with tongue at roof, release with 'SH'.",
            'S': "Keep tongue behind teeth for a clear 'S'.",
            'Z': "Add voice to the 'S' sound.",
            'NG': "Sound comes from back of throat.",
            'W': "Round your lips like saying 'oo'.",
            'Y': "Touch tongue to roof, slide to the next sound.",
            'V': "Touch upper teeth to lower lip, add voice.",
            'F': "Touch upper teeth to lower lip, blow air.",
        }
        
        return basic_tips.get(base, f"Practice the '{phoneme}' sound more carefully.")
    
    def _generate_summary(self, mistakes: List[Mistake]) -> str:
        """Generate one-line summary of all mistakes."""
        if not mistakes:
            return "Perfect pronunciation!"
        
        word_errors = len([m for m in mistakes if 'word' in m.type])
        sound_errors = len([m for m in mistakes if 'sound' in m.type or 'phoneme' in m.type])
        
        parts = []
        if word_errors:
            parts.append(f"{word_errors} word(s)")
        if sound_errors:
            parts.append(f"{sound_errors} sound(s)")
        
        return f"Work on: {', '.join(parts)}" if parts else "Minor improvements needed"


# Singleton
_mistake_detector = None


def get_mistake_detector() -> MistakeDetector:
    """Get singleton mistake detector instance."""
    global _mistake_detector
    if _mistake_detector is None:
        _mistake_detector = MistakeDetector()
    return _mistake_detector


def detect_mistakes(asr_result: Dict, phoneme_scores: List[Dict], expected_text: str) -> Dict:
    """Convenience function for detecting mistakes."""
    return get_mistake_detector().detect_mistakes(asr_result, phoneme_scores, expected_text)
