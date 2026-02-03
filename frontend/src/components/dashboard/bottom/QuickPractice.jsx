import { useNavigate } from 'react-router-dom';
import { Zap, Mic } from 'lucide-react';
import './QuickPractice.css';

export function QuickPractice() {
    const navigate = useNavigate();

    const handleStart = () => {
        navigate('/practice');
    };

    return (
        <div className="quick-practice">
            <div className="quick-practice__content">
                <span className="quick-practice__label">Quick Start</span>
                <h3 className="quick-practice__title">
                    Got 2 minutes? <br />
                    Complete a quick pronunciation drill.
                </h3>
                <button className="quick-practice__btn" onClick={handleStart}>
                    <Mic size={18} />
                    Start Session
                </button>
            </div>
            <Zap size={120} className="quick-practice__icon" />
        </div>
    );
}
