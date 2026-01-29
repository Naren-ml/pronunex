import { useState, useEffect } from 'react';
import { Sparkles, ChevronDown } from 'lucide-react';
import './InsightsPanel.css';

export function InsightsPanel({ children, isOpen: initialIsOpen = false, title = "Advanced Insights & AI Feedback" }) {
    const [isOpen, setIsOpen] = useState(initialIsOpen);

    useEffect(() => {
        setIsOpen(initialIsOpen);
    }, [initialIsOpen]);

    return (
        <div className={`insights-panel ${!isOpen ? 'insights-panel--collapsed' : ''}`}>
            <div className="insights-panel__header" onClick={() => setIsOpen(!isOpen)}>
                <h3 className="insights-panel__title">
                    <Sparkles size={18} className="text-secondary" />
                    {title}
                </h3>
                <ChevronDown size={20} className="insights-panel__toggle" />
            </div>

            <div className="insights-panel__content">
                {children}
            </div>
        </div>
    );
}
