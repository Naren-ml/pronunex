/**
 * DifficultyBadge Component
 * Displays sentence difficulty level with visual cues following EdTech scaffolding principles
 */

import { Sparkles, Zap, Trophy } from 'lucide-react';
import './DifficultyBadge.css';

const DIFFICULTY_CONFIG = {
    beginner: {
        label: 'Beginner',
        icon: Sparkles,
        description: 'Simple words, clear pronunciation'
    },
    intermediate: {
        label: 'Intermediate',
        icon: Zap,
        description: 'Compound sentences, stress patterns'
    },
    advanced: {
        label: 'Advanced',
        icon: Trophy,
        description: 'Complex prose, rhythm & elision'
    }
};

function DifficultyBadge({ level = 'beginner', showDescription = false }) {
    const config = DIFFICULTY_CONFIG[level] || DIFFICULTY_CONFIG.beginner;
    const IconComponent = config.icon;

    return (
        <div className={`difficulty-badge difficulty-badge--${level}`}>
            <IconComponent size={16} className="difficulty-badge__icon" />
            <span className="difficulty-badge__label">{config.label}</span>
            {showDescription && (
                <span className="difficulty-badge__description">{config.description}</span>
            )}
        </div>
    );
}

export default DifficultyBadge;
