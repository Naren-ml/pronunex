import { Bot, ArrowRight } from 'lucide-react';
import './AIRecommendations.css';

export function AIRecommendations({ assessment, currentSentence }) {
    if (!assessment) return null;

    // Logic to generate recommendation text
    const getRecommendationText = () => {
        if (assessment.overall_score >= 80) {
            return `Great job! Your intonation matched the reference very well. Try increasing your speaking speed slightly to match natural conversation flow.`;
        }

        const weakPhonemes = assessment.phoneme_scores
            ?.filter(p => p.score < 60)
            .map(p => `/${p.phoneme}/`)
            .join(', ');

        if (weakPhonemes) {
            return `Focus on articulating the ${weakPhonemes} sounds more clearly. Your tongue placement seems slightly off for these specific phonemes.`;
        }

        return "Good effort! Try to emphasize the stressed syllables more to improve your rhythm.";
    };

    return (
        <div className="ai-recommendations">
            <header className="ai-recommendations__header">
                <Bot size={18} className="text-primary" />
                <h4 className="ai-recommendations__title">AI Coach</h4>
            </header>

            <p className="ai-recommendations__text">
                {getRecommendationText()}
            </p>

            <div className="ai-recommendations__actions">
                <button className="ai-recommendations__action-btn">
                    Practice Similar Words
                </button>
            </div>
        </div>
    );
}
