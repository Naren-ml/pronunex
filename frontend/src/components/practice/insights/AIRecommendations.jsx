/**
 * AIRecommendations Component
 * 
 * Displays AI-powered pronunciation feedback and tips.
 * Updated to use new NLP pipeline data (llm_feedback, mistakes, etc.)
 */

import { Bot, ArrowRight, Lightbulb, Target } from 'lucide-react';
import './AIRecommendations.css';

export function AIRecommendations({ assessment, currentSentence }) {
    if (!assessment) return null;

    // Get feedback from the new API format
    const feedback = assessment.llm_feedback || {};
    const mistakes = assessment.mistakes || {};
    const mistakeFeedback = mistakes.feedback || {};

    // Determine recommendation text from various sources
    const getRecommendationText = () => {
        // Priority 1: LLM-generated summary
        if (feedback.summary) {
            return feedback.summary;
        }

        // Priority 2: Mistake feedback message
        if (mistakeFeedback.message) {
            return mistakeFeedback.message;
        }

        // Priority 3: Score-based fallback
        if (assessment.overall_score >= 0.8) {
            return "Great job! Your pronunciation matched the reference very well. Try increasing your speaking speed slightly to match natural conversation flow.";
        }

        // Priority 4: Weak phoneme based feedback
        const weakPhonemes = assessment.phoneme_scores
            ?.filter(p => p.score < 0.6)
            .map(p => `/${p.phoneme}/`)
            .join(', ');

        if (weakPhonemes) {
            return `Focus on articulating the ${weakPhonemes} sounds more clearly. Your tongue placement seems slightly off for these specific phonemes.`;
        }

        return "Good effort! Try to emphasize the stressed syllables more to improve your rhythm.";
    };

    // Collect all tips from various sources
    const getAllTips = () => {
        const tips = [];

        // Helper to extract tip text from object or string
        const extractTipText = (tip) => {
            if (typeof tip === 'string') return tip;
            if (tip && typeof tip === 'object') {
                // Try common properties for tip text
                const content = tip.suggestion || tip.text || tip.message || tip.expected;

                // If the content is still an object (nested), try to extract from it or stringify
                if (content && typeof content === 'object') {
                    return content.text || content.message || JSON.stringify(content);
                }

                return content || null;
            }
            return null;
        };

        // From LLM feedback
        if (feedback.phoneme_tips && Array.isArray(feedback.phoneme_tips)) {
            feedback.phoneme_tips.forEach(tip => {
                const text = extractTipText(tip);
                if (text) tips.push(text);
            });
        }

        // From mistake feedback tips
        if (mistakeFeedback.tips && Array.isArray(mistakeFeedback.tips)) {
            mistakeFeedback.tips.forEach(tip => {
                const text = extractTipText(tip);
                if (text) tips.push(text);
            });
        }

        // From individual mistakes with suggestions
        if (mistakes.mistakes && Array.isArray(mistakes.mistakes)) {
            mistakes.mistakes.forEach(m => {
                if (m.suggestion) tips.push(m.suggestion);
            });
        }

        // Remove duplicates and limit
        return [...new Set(tips)].slice(0, 4);
    };

    const tips = getAllTips();

    return (
        <div className="ai-recommendations">
            <header className="ai-recommendations__header">
                <Bot size={18} className="text-primary" />
                <h4 className="ai-recommendations__title">AI Coach</h4>
            </header>

            <p className="ai-recommendations__text">
                {getRecommendationText()}
            </p>

            {/* Encouragement message */}
            {feedback.encouragement && (
                <p className="ai-recommendations__encouragement">
                    {feedback.encouragement}
                </p>
            )}

            {/* Tips list */}
            {tips.length > 0 && (
                <div className="ai-recommendations__tips">
                    <div className="ai-recommendations__tips-header">
                        <Lightbulb size={14} />
                        <span>Quick Tips</span>
                    </div>
                    <ul className="ai-recommendations__tips-list">
                        {tips.map((tip, idx) => (
                            <li key={idx}>{tip}</li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Transcription comparison (if available) */}
            {assessment.transcribed && (
                <div className="ai-recommendations__transcription">
                    <Target size={14} />
                    <span>We heard: "{assessment.transcribed}"</span>
                </div>
            )}

            <div className="ai-recommendations__actions">
                <button className="ai-recommendations__action-btn">
                    Practice Similar Words
                    <ArrowRight size={14} />
                </button>
            </div>
        </div>
    );
}

export default AIRecommendations;
