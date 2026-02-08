/**
 * MistakePanel Component
 * 
 * Displays detailed mistake detection results from the NLP pipeline.
 * Shows word-level errors, phoneme errors, and AI-generated tips.
 */

import { AlertTriangle, CheckCircle, Target, Lightbulb, X, AlertCircle } from 'lucide-react';
import './MistakePanel.css';

export function MistakePanel({ mistakes, transcribed, expectedText }) {
    if (!mistakes) return null;

    const { has_mistakes, mistake_count, mistakes: mistakeList = [], feedback = {} } = mistakes;

    // No mistakes - show success
    if (!has_mistakes || mistake_count === 0) {
        return (
            <div className="mistake-panel mistake-panel--success">
                <div className="mistake-panel__header">
                    <CheckCircle size={20} className="mistake-panel__icon--success" />
                    <span className="mistake-panel__title">Perfect Match!</span>
                </div>
                <p className="mistake-panel__message">
                    You pronounced every word correctly. Great job!
                </p>
            </div>
        );
    }

    // Status-based styling
    const getStatusClass = () => {
        const status = feedback.status || 'needs_improvement';
        if (status === 'almost_perfect') return 'mistake-panel--warning';
        if (status === 'good') return 'mistake-panel--info';
        return 'mistake-panel--error';
    };

    const getStatusIcon = () => {
        const status = feedback.status || 'needs_improvement';
        if (status === 'almost_perfect') return <AlertCircle size={20} />;
        if (status === 'good') return <Target size={20} />;
        return <AlertTriangle size={20} />;
    };

    return (
        <div className={`mistake-panel ${getStatusClass()}`}>
            {/* Header */}
            <div className="mistake-panel__header">
                {getStatusIcon()}
                <span className="mistake-panel__title">
                    {feedback.message || `${mistake_count} area(s) to improve`}
                </span>
            </div>

            {/* What you said vs expected */}
            {transcribed && (
                <div className="mistake-panel__transcription">
                    <div className="mistake-panel__transcription-row">
                        <span className="mistake-panel__label">You said:</span>
                        <span className="mistake-panel__value mistake-panel__value--user">
                            "{transcribed}"
                        </span>
                    </div>
                    <div className="mistake-panel__transcription-row">
                        <span className="mistake-panel__label">Expected:</span>
                        <span className="mistake-panel__value mistake-panel__value--expected">
                            "{expectedText}"
                        </span>
                    </div>
                </div>
            )}

            {/* Mistake List */}
            {mistakeList.length > 0 && (
                <div className="mistake-panel__list">
                    {mistakeList.slice(0, 5).map((mistake, idx) => (
                        <MistakeItem key={idx} mistake={mistake} />
                    ))}
                    {mistakeList.length > 5 && (
                        <p className="mistake-panel__more">
                            +{mistakeList.length - 5} more areas to work on
                        </p>
                    )}
                </div>
            )}

            {/* AI Tips */}
            {feedback.tips && feedback.tips.length > 0 && (
                <div className="mistake-panel__tips">
                    <div className="mistake-panel__tips-header">
                        <Lightbulb size={16} />
                        <span>Tips for Improvement</span>
                    </div>
                    <ul className="mistake-panel__tips-list">
                        {feedback.tips.map((tip, idx) => {
                            // Handle both string and object tips
                            // Handle both string and object tips safely
                            let tipText = typeof tip === 'string' ? tip : null;

                            if (!tipText && typeof tip === 'object') {
                                const content = tip?.suggestion || tip?.text || tip?.message || tip?.expected;
                                if (typeof content === 'object') {
                                    tipText = content?.text || content?.message || JSON.stringify(content);
                                } else {
                                    tipText = content;
                                }

                                if (!tipText) tipText = JSON.stringify(tip);
                            }
                            return <li key={idx}>{tipText}</li>;
                        })}
                    </ul>
                </div>
            )}
        </div>
    );
}

function MistakeItem({ mistake }) {
    const getIcon = () => {
        switch (mistake.type) {
            case 'missing_word':
            case 'missing_sound':
                return <X size={14} className="mistake-item__icon--missing" />;
            case 'wrong_word':
                return <AlertTriangle size={14} className="mistake-item__icon--wrong" />;
            case 'weak_phoneme':
                return <Target size={14} className="mistake-item__icon--weak" />;
            default:
                return <AlertCircle size={14} />;
        }
    };

    const getDescription = () => {
        if (mistake.suggestion) {
            if (typeof mistake.suggestion === 'object') {
                return mistake.suggestion.text || mistake.suggestion.message || JSON.stringify(mistake.suggestion);
            }
            return mistake.suggestion;
        }

        switch (mistake.type) {
            case 'missing_sound':
                return `Missing "${mistake.expected?.slice(-1)}" at the end of "${mistake.actual}"`;
            case 'missing_word':
                return `Skipped the word "${mistake.expected}"`;
            case 'wrong_word':
                return `Said "${mistake.actual}" instead of "${mistake.expected}"`;
            case 'weak_phoneme':
                return `The /${mistake.phoneme}/ sound in "${mistake.word}" needs work`;
            default:
                return `Issue with "${mistake.expected || mistake.word}"`;
        }
    };

    return (
        <div className={`mistake-item mistake-item--${mistake.severity || 'minor'}`}>
            <div className="mistake-item__icon">{getIcon()}</div>
            <div className="mistake-item__content">
                <span className="mistake-item__description">{getDescription()}</span>
                {mistake.score !== undefined && (
                    <span className="mistake-item__score">
                        {Math.round(mistake.score * 100)}% accuracy
                    </span>
                )}
            </div>
        </div>
    );
}

/**
 * ContentMismatchError Component
 * 
 * Shown when user says completely wrong sentence.
 */
export function ContentMismatchError({ error, onRetry }) {
    return (
        <div className="content-mismatch-error">
            <div className="content-mismatch-error__icon">
                <AlertTriangle size={48} />
            </div>
            <h3 className="content-mismatch-error__title">
                Different Sentence Detected
            </h3>
            <p className="content-mismatch-error__message">
                {error.message || "It looks like you said something different from the expected sentence."}
            </p>

            <div className="content-mismatch-error__comparison">
                <div className="content-mismatch-error__row">
                    <span className="content-mismatch-error__label">You said:</span>
                    <span className="content-mismatch-error__value content-mismatch-error__value--user">
                        "{error.transcribed || 'Unable to transcribe'}"
                    </span>
                </div>
                <div className="content-mismatch-error__row">
                    <span className="content-mismatch-error__label">Expected:</span>
                    <span className="content-mismatch-error__value content-mismatch-error__value--expected">
                        "{error.expected}"
                    </span>
                </div>
                <div className="content-mismatch-error__similarity">
                    Match: {Math.round((error.similarity || 0) * 100)}%
                </div>
            </div>

            <p className="content-mismatch-error__suggestion">
                {error.suggestion || "Please try again and say the exact sentence shown above."}
            </p>

            <button
                type="button"
                className="content-mismatch-error__retry-btn"
                onClick={onRetry}
            >
                Try Again
            </button>
        </div>
    );
}

/**
 * UnscorableError Component
 * 
 * Shown when technical issues prevent scoring.
 */
export function UnscorableError({ error, onRetry }) {
    return (
        <div className="unscorable-error">
            <div className="unscorable-error__icon">
                <AlertCircle size={48} />
            </div>
            <h3 className="unscorable-error__title">
                Could Not Analyze Audio
            </h3>
            <p className="unscorable-error__message">
                {error.message || "We couldn't process your recording properly."}
            </p>

            <p className="unscorable-error__suggestion">
                {error.suggestion || "Please speak more clearly and try again."}
            </p>

            <button
                type="button"
                className="unscorable-error__retry-btn"
                onClick={onRetry}
            >
                Try Again
            </button>
        </div>
    );
}

export default MistakePanel;
