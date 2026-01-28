/**
 * RecommendationCard Component
 * Adaptive "Next Step" recommendation based on assessment results
 */

import { Lightbulb, ArrowRight, BookOpen, Repeat } from 'lucide-react';
import './RecommendationCard.css';

function RecommendationCard({ weakPhonemes = [], overallScore = 0, onAction }) {
    const getRecommendation = () => {
        if (overallScore >= 0.9) {
            return {
                type: 'advance',
                icon: ArrowRight,
                title: 'Ready to Level Up',
                message: 'Excellent work. Try a more challenging sentence.',
                action: 'Next Challenge'
            };
        }

        if (weakPhonemes.length > 0) {
            const phoneme = weakPhonemes[0];
            return {
                type: 'practice',
                icon: Repeat,
                title: `Focus on /${phoneme}/`,
                message: `Practice this sound with a targeted exercise.`,
                action: 'Practice Phoneme'
            };
        }

        if (overallScore >= 0.7) {
            return {
                type: 'retry',
                icon: Repeat,
                title: 'Almost There',
                message: 'Try recording again for a higher score.',
                action: 'Try Again'
            };
        }

        return {
            type: 'learn',
            icon: BookOpen,
            title: 'Review the Basics',
            message: 'Listen to the example and study the pronunciation.',
            action: 'Learn More'
        };
    };

    const recommendation = getRecommendation();
    const IconComponent = recommendation.icon;

    return (
        <div className={`recommendation-card recommendation-card--${recommendation.type}`}>
            <div className="recommendation-card__header">
                <Lightbulb size={16} className="recommendation-card__bulb" />
                <span className="recommendation-card__label">Next Step</span>
            </div>
            <div className="recommendation-card__content">
                <IconComponent size={24} className="recommendation-card__icon" />
                <div className="recommendation-card__text">
                    <h4 className="recommendation-card__title">{recommendation.title}</h4>
                    <p className="recommendation-card__message">{recommendation.message}</p>
                </div>
            </div>
            {onAction && (
                <button
                    className="recommendation-card__action"
                    type="button"
                    onClick={() => onAction(recommendation.type)}
                >
                    {recommendation.action}
                    <ArrowRight size={16} />
                </button>
            )}
        </div>
    );
}

export default RecommendationCard;
