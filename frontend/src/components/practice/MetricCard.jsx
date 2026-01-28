/**
 * MetricCard Component
 * Glassmorphic card for displaying individual practice metrics
 */

import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import './MetricCard.css';

function MetricCard({
    label,
    value,
    unit = '%',
    icon: Icon,
    trend = null,  // 'up' | 'down' | 'neutral' | null
    trendValue = null,
    size = 'default'  // 'default' | 'compact'
}) {
    const getTrendIcon = () => {
        if (!trend) return null;
        switch (trend) {
            case 'up':
                return <TrendingUp size={14} className="metric-card__trend-icon metric-card__trend-icon--up" />;
            case 'down':
                return <TrendingDown size={14} className="metric-card__trend-icon metric-card__trend-icon--down" />;
            default:
                return <Minus size={14} className="metric-card__trend-icon metric-card__trend-icon--neutral" />;
        }
    };

    return (
        <div className={`metric-card metric-card--${size}`}>
            <div className="metric-card__header">
                {Icon && <Icon size={18} className="metric-card__icon" />}
                <span className="metric-card__label">{label}</span>
            </div>
            <div className="metric-card__value-container">
                <span className="metric-card__value">{value}</span>
                {unit && <span className="metric-card__unit">{unit}</span>}
            </div>
            {trend && (
                <div className={`metric-card__trend metric-card__trend--${trend}`}>
                    {getTrendIcon()}
                    {trendValue && <span className="metric-card__trend-value">{trendValue}</span>}
                </div>
            )}
        </div>
    );
}

export default MetricCard;
