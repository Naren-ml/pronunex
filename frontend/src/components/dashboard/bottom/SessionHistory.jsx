import { useNavigate } from 'react-router-dom';
import { History, Clock, ArrowRight } from 'lucide-react';
import { useApi } from '../../../hooks/useApi';
import { ENDPOINTS } from '../../../api/endpoints';
import { Spinner } from '../../Loader';
import './SessionHistory.css';

export function SessionHistory() {
    const navigate = useNavigate();
    // Use ANALYTICS.HISTORY endpoint - same as Progress page
    const { data: history, isLoading } = useApi(`${ENDPOINTS.ANALYTICS.HISTORY}?days=30`);

    // Process history data similar to Progress page
    const recentSessions = (() => {
        if (!history) return [];
        const historyArray = Array.isArray(history) ? history : (history.results || history.data || []);
        if (!Array.isArray(historyArray)) return [];

        return historyArray.slice(0, 3).map(item => ({
            date: item.date,
            score: item.average_score || 0,
            attempts: item.attempts_count || item.attempts || 0,
            duration: item.total_practice_minutes || item.duration || 0,
        }));
    })();

    // Helper to format date
    const formatDate = (dateString) => {
        const date = new Date(dateString);
        return new Intl.DateTimeFormat('en-US', {
            month: 'short',
            day: 'numeric',
        }).format(date);
    };

    // Helper for score badge class
    const getScoreClass = (score) => {
        const percentage = score * 100;
        if (percentage >= 80) return 'session-history__score-badge--high';
        if (percentage >= 60) return 'session-history__score-badge--medium';
        return 'session-history__score-badge--low';
    };

    if (isLoading) {
        return (
            <div className="session-history">
                <Spinner size="sm" />
            </div>
        );
    }

    return (
        <div className="session-history">
            <header className="session-history__header">
                <h3 className="session-history__title">
                    <History size={18} />
                    Recent Sessions
                </h3>
                <span
                    className="session-history__link"
                    onClick={() => navigate('/progress')}
                    style={{ cursor: 'pointer' }}
                >
                    View All
                </span>
            </header>

            <div className="session-history__list">
                {recentSessions.length > 0 ? (
                    recentSessions.map((session, idx) => (
                        <div key={idx} className="session-history__item">
                            <div className="session-history__info">
                                <span className="session-history__date">
                                    {formatDate(session.date)}
                                </span>
                                <div className="session-history__meta">
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                                        <Clock size={12} />
                                        {session.attempts} attempts
                                    </span>
                                </div>
                            </div>

                            <div className="session-history__score">
                                <span className={`session-history__score-badge ${getScoreClass(session.score)}`}>
                                    {Math.round(session.score * 100)}%
                                </span>
                                <ArrowRight size={14} className="text-secondary" />
                            </div>
                        </div>
                    ))
                ) : (
                    <div className="session-history__empty">
                        <p>No recent sessions found.</p>
                    </div>
                )}
            </div>
        </div>
    );
}
