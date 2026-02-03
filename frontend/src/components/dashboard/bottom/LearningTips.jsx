import { useState } from 'react';
import { Lightbulb, ChevronLeft, ChevronRight } from 'lucide-react';
import './LearningTips.css';

const TIPS = [
    {
        title: "Tongue Placement for /TH/",
        text: "Place the tip of your tongue gently between your upper and lower teeth. Blow air through to create the friction sound."
    },
    {
        title: "The 'R' Coloring",
        text: "For the American /r/, try curling the tip of your tongue back slightly without touching the roof of your mouth."
    },
    {
        title: "Breath Control",
        text: "Take a deep breath before starting a long sentence. Pause at commas to replenish your air supply naturally."
    }
];

export function LearningTips() {
    const [currentIndex, setCurrentIndex] = useState(0);

    const handleNext = () => {
        setCurrentIndex((prev) => (prev + 1) % TIPS.length);
    };

    const handlePrev = () => {
        setCurrentIndex((prev) => (prev - 1 + TIPS.length) % TIPS.length);
    };

    const currentTip = TIPS[currentIndex];

    return (
        <div className="learning-tips">
            <header className="learning-tips__header">
                <h3 className="learning-tips__title">
                    <Lightbulb size={18} className="text-warning" />
                    Daily Tips
                </h3>
                <div className="learning-tips__controls">
                    <button onClick={handlePrev} className="learning-tips__control-btn">
                        <ChevronLeft size={16} />
                    </button>
                    <button onClick={handleNext} className="learning-tips__control-btn">
                        <ChevronRight size={16} />
                    </button>
                </div>
            </header>

            <div className="learning-tips__content">
                <div className="learning-tips__icon-box">
                    <Lightbulb size={24} />
                </div>
                <div className="learning-tips__text">
                    <h4>{currentTip.title}</h4>
                    <p>{currentTip.text}</p>
                </div>
            </div>
        </div>
    );
}
