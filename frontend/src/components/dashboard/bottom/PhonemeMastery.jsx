import { useNavigate } from 'react-router-dom';
import { BookOpen } from 'lucide-react';
import { useApi } from '../../../hooks/useApi';
import { ENDPOINTS } from '../../../api/endpoints';
import { Spinner } from '../../Loader';
import './PhonemeMastery.css';

export function PhonemeMastery() {
    const navigate = useNavigate();
    const { data: stats, isLoading } = useApi(ENDPOINTS.ANALYTICS.PHONEME_STATS);

    // Mock data if endpoint returns nothing or error (for robust UI dev)
    const phonemeData = stats?.length > 0 ? stats : [
        { phoneme: 'th', score: 0.85 },
        { phoneme: 'r', score: 0.45 },
        { phoneme: 'l', score: 0.62 },
        { phoneme: 'ae', score: 0.92 },
        { phoneme: 'sh', score: 0.78 },
        { phoneme: 'ch', score: 0.35 },
        { phoneme: 'v', score: 0.88 },
        { phoneme: 'f', score: 0.95 },
    ];

    const getScoreColor = (score) => {
        if (score >= 0.8) return '#10b981'; // Success
        if (score >= 0.5) return '#f59e0b'; // Warning
        return '#ef4444'; // Danger
    };

    if (isLoading) return <div className="phoneme-mastery"><Spinner size="sm" /></div>;

    return (
        <div className="phoneme-mastery">
            <header className="phoneme-mastery__header">
                <h3 className="phoneme-mastery__title">
                    <BookOpen size={18} />
                    Phoneme Mastery
                </h3>
                <div className="phoneme-mastery__legend">
                    <div className="phoneme-mastery__legend-item">
                        <span className="phoneme-mastery__dot bg-success"></span> Strong
                    </div>
                    <div className="phoneme-mastery__legend-item">
                        <span className="phoneme-mastery__dot bg-warning"></span> Average
                    </div>
                    <div className="phoneme-mastery__legend-item">
                        <span className="phoneme-mastery__dot bg-danger"></span> Weak
                    </div>
                </div>
            </header>

            <div className="phoneme-mastery__grid">
                {phonemeData.slice(0, 8).map((item, idx) => (
                    <div
                        key={idx}
                        className="phoneme-mastery__item"
                        onClick={() => navigate(`/phonemes/${item.phoneme}`)}
                    >
                        <span className="phoneme-mastery__symbol">/{item.phoneme}/</span>
                        <div className="phoneme-mastery__bar">
                            <div
                                className="phoneme-mastery__fill"
                                style={{
                                    width: `${item.score * 100}%`,
                                    background: getScoreColor(item.score)
                                }}
                            ></div>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
}
