/**
 * Progress Page - Comprehensive Analytics Dashboard
 * Displays user's pronunciation improvement journey with detailed visualizations
 */

import { useState, useMemo, useRef, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
    AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, PieChart, Pie
} from 'recharts';
import {
    TrendingUp, TrendingDown, Minus, Target, Award, Clock, Flame,
    ChevronRight, Calendar, Mic, BookOpen, Zap, Filter
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { ENDPOINTS } from '../api/endpoints';
import { Card } from '../components/Card';
import { Spinner } from '../components/Loader';
import { ErrorState } from '../components/ErrorState';
import { NoProgress } from '../components/EmptyState';
import './Progress.css';

// Chart color palette - uses teal/emerald theme
const CHART_COLORS = {
    primary: '#14b8a6',
    primaryLight: '#5eead4',
    success: '#10b981',
    warning: '#f59e0b',
    error: '#ef4444',
    neutral: '#64748b',
};

// Sparkline Component
function Sparkline({ data = [], color = '#14b8a6', height = 36 }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || data.length === 0) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const h = canvas.height;
        const padding = 4;

        ctx.clearRect(0, 0, width, h);

        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const range = max - min || 1;

        const points = data.map((value, index) => ({
            x: padding + (index / (data.length - 1)) * (width - padding * 2),
            y: h - padding - ((value - min) / range) * (h - padding * 2)
        }));

        // Gradient fill
        const gradient = ctx.createLinearGradient(0, 0, 0, h);
        gradient.addColorStop(0, color + '30');
        gradient.addColorStop(1, color + '05');

        ctx.beginPath();
        ctx.moveTo(points[0].x, h);
        ctx.lineTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xMid = (points[i].x + points[i + 1].x) / 2;
            const yMid = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xMid, yMid);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.lineTo(points[points.length - 1].x, h);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Line
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xMid = (points[i].x + points[i + 1].x) / 2;
            const yMid = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xMid, yMid);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.lineCap = 'round';
        ctx.stroke();
    }, [data, color, height]);

    return (
        <canvas
            ref={canvasRef}
            width={120}
            height={height}
            className="progress__sparkline"
        />
    );
}

// Stats Card Component
function StatsCard({ icon: Icon, label, value, sparkData, trend, color = CHART_COLORS.primary }) {
    const trendClass = trend > 0 ? 'up' : trend < 0 ? 'down' : 'neutral';

    return (
        <div className="progress__stat-card">
            <div className="progress__stat-header">
                <span className="progress__stat-label">{label}</span>
                <div className="progress__stat-icon" style={{ backgroundColor: color + '20', color }}>
                    <Icon size={18} />
                </div>
            </div>
            <span className="progress__stat-value">{value}</span>
            {sparkData && sparkData.length > 0 && (
                <Sparkline data={sparkData} color={color} />
            )}
            {trend !== undefined && trend !== null && (
                <div className={`progress__stat-trend progress__stat-trend--${trendClass}`}>
                    {trend > 0 && <TrendingUp size={14} />}
                    {trend < 0 && <TrendingDown size={14} />}
                    {trend === 0 && <Minus size={14} />}
                    <span>{trend > 0 ? '+' : ''}{trend}% this week</span>
                </div>
            )}
        </div>
    );
}

// Phoneme Mastery Bar Component
function PhonemeMasteryBar({ phoneme, symbol, score, attempts }) {
    const percentage = Math.round(score * 100);
    const level = score >= 0.85 ? 'mastered' : score >= 0.7 ? 'proficient' : score >= 0.5 ? 'developing' : 'needs-work';
    const levelLabel = score >= 0.85 ? 'Mastered' : score >= 0.7 ? 'Proficient' : score >= 0.5 ? 'Developing' : 'Needs Work';

    return (
        <div className="progress__mastery-item">
            <div className="progress__mastery-header">
                <div className="progress__mastery-phoneme">
                    <span className="progress__mastery-symbol">/{symbol || phoneme}/</span>
                    <span className="progress__mastery-arpabet">{phoneme}</span>
                </div>
                <span className={`progress__mastery-badge progress__mastery-badge--${level}`}>
                    {levelLabel}
                </span>
            </div>
            <div className="progress__mastery-bar-container">
                <div
                    className={`progress__mastery-bar progress__mastery-bar--${level}`}
                    style={{ width: `${percentage}%` }}
                />
            </div>
            <div className="progress__mastery-meta">
                <span className="progress__mastery-score">{percentage}%</span>
                <span className="progress__mastery-attempts">{attempts} attempts</span>
            </div>
        </div>
    );
}

// Session History Item Component
function SessionHistoryItem({ date, score, attempts, duration }) {
    const formattedDate = new Date(date).toLocaleDateString('en-US', {
        weekday: 'short',
        month: 'short',
        day: 'numeric'
    });
    const scorePercent = Math.round((score || 0) * 100);
    const scoreClass = scorePercent >= 80 ? 'excellent' : scorePercent >= 60 ? 'good' : 'needs-work';

    return (
        <div className="progress__history-item">
            <div className="progress__history-date">
                <Calendar size={16} />
                <span>{formattedDate}</span>
            </div>
            <div className="progress__history-details">
                <span className={`progress__history-score progress__history-score--${scoreClass}`}>
                    {scorePercent}%
                </span>
                <span className="progress__history-attempts">{attempts} attempts</span>
                {duration && (
                    <span className="progress__history-duration">
                        <Clock size={14} />
                        {Math.round(duration)}m
                    </span>
                )}
            </div>
        </div>
    );
}

// Period Selector Component
function PeriodSelector({ value, onChange }) {
    const periods = [
        { key: '7', label: '7 Days' },
        { key: '30', label: '30 Days' },
        { key: '90', label: '90 Days' },
    ];

    return (
        <div className="progress__period-selector">
            {periods.map(p => (
                <button
                    key={p.key}
                    className={`progress__period-btn ${value === p.key ? 'progress__period-btn--active' : ''}`}
                    onClick={() => onChange(p.key)}
                >
                    {p.label}
                </button>
            ))}
        </div>
    );
}

// Custom Tooltip for Charts
function CustomTooltip({ active, payload, label }) {
    if (!active || !payload || !payload.length) return null;

    return (
        <div className="progress__tooltip">
            <p className="progress__tooltip-label">{label}</p>
            {payload.map((entry, index) => (
                <p key={index} className="progress__tooltip-value" style={{ color: entry.color }}>
                    {entry.name}: {entry.value}%
                </p>
            ))}
        </div>
    );
}

export function Progress() {
    const navigate = useNavigate();
    const [period, setPeriod] = useState('30');

    // Fetch all required data
    const { data: progressData, isLoading: progressLoading, error: progressError, refetch: refetchProgress } =
        useApi(ENDPOINTS.ANALYTICS.PROGRESS);
    const { data: history, isLoading: historyLoading, error: historyError, refetch: refetchHistory } =
        useApi(`${ENDPOINTS.ANALYTICS.HISTORY}?days=${period}`);
    const { data: phonemeStats, isLoading: phonemesLoading, error: phonemesError, refetch: refetchPhonemes } =
        useApi(ENDPOINTS.ANALYTICS.PHONEME_STATS);

    const isLoading = progressLoading || historyLoading || phonemesLoading;
    const error = progressError || historyError || phonemesError;

    // Process history data for chart
    const chartData = useMemo(() => {
        if (!history) return [];
        const historyArray = Array.isArray(history) ? history : (history.results || history.data || []);
        if (!Array.isArray(historyArray)) return [];

        return historyArray
            .map((item) => ({
                date: new Date(item.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                score: Math.round((item.average_score || 0) * 100),
                attempts: item.attempts_count || item.attempts || 0,
            }))
            .reverse();
    }, [history]);

    // Process phoneme data for mastery section - use data from multiple sources
    const phonemeData = useMemo(() => {
        // Try phoneme-stats endpoint first
        if (phonemeStats && phonemeStats.by_type) {
            const byType = phonemeStats.by_type || {};
            const allPhonemes = [];

            Object.values(byType).forEach(phonemes => {
                if (Array.isArray(phonemes)) {
                    allPhonemes.push(...phonemes);
                }
            });

            if (allPhonemes.length > 0) {
                const sorted = allPhonemes.sort((a, b) => (a.current_score || 0) - (b.current_score || 0));
                const weak = sorted.filter(p => (p.current_score || 0) < 0.7);
                const strong = sorted.filter(p => (p.current_score || 0) >= 0.85);
                return { all: sorted, weak, strong };
            }
        }

        // Fallback to phoneme_progress from progressData
        if (progressData && progressData.phoneme_progress && Array.isArray(progressData.phoneme_progress)) {
            const allPhonemes = progressData.phoneme_progress.map(p => ({
                phoneme: p.phoneme?.arpabet || p.phoneme_arpabet || p.phoneme,
                symbol: p.phoneme?.symbol || p.phoneme_symbol || p.symbol,
                current_score: p.current_score || 0,
                attempts: p.attempts_count || p.attempts || 0,
                best_score: p.best_score || 0,
            }));

            const sorted = allPhonemes.sort((a, b) => (a.current_score || 0) - (b.current_score || 0));
            const weak = sorted.filter(p => (p.current_score || 0) < 0.7);
            const strong = sorted.filter(p => (p.current_score || 0) >= 0.85);
            return { all: sorted, weak, strong };
        }

        // Fallback to weak/strong phonemes lists from progressData
        if (progressData) {
            const weak = (progressData.current_weak_phonemes || []).map(p =>
                typeof p === 'string' ? { phoneme: p, symbol: p, current_score: 0.5, attempts: 0 } : p
            );
            const strong = (progressData.current_strong_phonemes || []).map(p =>
                typeof p === 'string' ? { phoneme: p, symbol: p, current_score: 0.9, attempts: 0 } : p
            );
            return { all: [...weak, ...strong], weak, strong };
        }

        return { all: [], weak: [], strong: [] };
    }, [phonemeStats, progressData]);

    // Calculate trend from chart data
    const trend = useMemo(() => {
        if (!chartData || chartData.length < 2) return null;
        const recent = chartData.slice(-7);
        if (recent.length < 2) return null;

        const first = recent[0].score;
        const last = recent[recent.length - 1].score;
        const diff = last - first;

        return {
            direction: diff > 0 ? 'up' : diff < 0 ? 'down' : 'neutral',
            value: Math.abs(diff),
        };
    }, [chartData]);

    // Normalize progress data
    const stats = useMemo(() => {
        if (!progressData) return null;

        return {
            totalAttempts: progressData.total_attempts || 0,
            totalSessions: progressData.total_sessions || 0,
            averageScore: progressData.overall_average_score || 0,
            practiceMinutes: progressData.total_practice_minutes || 0,
            streak: progressData.streak || { current_streak: 0, longest_streak: 0 },
            weakPhonemes: progressData.current_weak_phonemes || [],
            strongPhonemes: progressData.current_strong_phonemes || [],
            scoreTrend: progressData.score_trend || 'insufficient_data',
        };
    }, [progressData]);

    // Generate sparkline data from history
    const sparklineData = useMemo(() => {
        if (!chartData || chartData.length === 0) return [];
        return chartData.map(d => d.score);
    }, [chartData]);

    // Process session history from multiple sources
    const sessionHistory = useMemo(() => {
        // First try to use history data
        if (history) {
            const historyArray = Array.isArray(history) ? history : (history.results || history.data || []);
            if (Array.isArray(historyArray) && historyArray.length > 0) {
                return historyArray.slice(0, 10).map(item => ({
                    date: item.date,
                    score: item.average_score || 0,
                    attempts: item.attempts_count || item.attempts || 0,
                    duration: item.total_practice_minutes || item.duration || 0,
                }));
            }
        }

        // Fallback to recent_progress from progressData
        if (progressData && progressData.recent_progress && Array.isArray(progressData.recent_progress)) {
            return progressData.recent_progress.slice(0, 10).map(item => ({
                date: item.date,
                score: item.average_score || 0,
                attempts: item.attempts_count || item.attempts || 0,
                duration: item.total_practice_minutes || item.duration || 0,
            }));
        }

        return [];
    }, [history, progressData]);

    // Handle period change
    const handlePeriodChange = (newPeriod) => {
        setPeriod(newPeriod);
    };

    // Retry all data fetches
    const handleRetry = () => {
        refetchProgress();
        refetchHistory();
        refetchPhonemes();
    };

    if (isLoading) {
        return (
            <div className="progress-loading">
                <Spinner size="lg" />
                <p>Loading your progress...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="progress-error">
                <ErrorState
                    icon="server"
                    title="Failed to load progress"
                    message="We could not load your progress data. Please try again."
                    onRetry={handleRetry}
                />
            </div>
        );
    }

    // Show empty state if no data
    const hasData = stats && stats.totalAttempts > 0;

    if (!hasData) {
        return (
            <div className="progress-empty">
                <NoProgress onStart={() => navigate('/practice')} />
            </div>
        );
    }

    return (
        <div className="progress">
            {/* Header */}
            <header className="progress__header">
                <div className="progress__header-content">
                    <h1 className="progress__title">Your Progress</h1>
                    <p className="progress__subtitle">
                        Track your pronunciation improvement journey
                    </p>
                </div>
                {trend && (
                    <div className={`progress__trend progress__trend--${trend.direction}`}>
                        {trend.direction === 'up' && <TrendingUp size={20} />}
                        {trend.direction === 'down' && <TrendingDown size={20} />}
                        {trend.direction === 'neutral' && <Minus size={20} />}
                        <span>
                            {trend.direction === 'up' && `+${trend.value}% this week`}
                            {trend.direction === 'down' && `-${trend.value}% this week`}
                            {trend.direction === 'neutral' && 'No change this week'}
                        </span>
                    </div>
                )}
            </header>

            {/* Stats Overview Grid */}
            <section className="progress__stats-grid">
                <StatsCard
                    icon={Target}
                    label="Total Attempts"
                    value={stats.totalAttempts}
                    sparkData={sparklineData}
                    color={CHART_COLORS.primary}
                />
                <StatsCard
                    icon={Award}
                    label="Average Score"
                    value={`${Math.round(stats.averageScore * 100)}%`}
                    sparkData={sparklineData}
                    trend={trend?.direction === 'up' ? trend.value : trend?.direction === 'down' ? -trend.value : 0}
                    color={CHART_COLORS.success}
                />
                <StatsCard
                    icon={Clock}
                    label="Practice Time"
                    value={`${Math.round(stats.practiceMinutes)}m`}
                    color={CHART_COLORS.neutral}
                />
                <StatsCard
                    icon={Flame}
                    label="Current Streak"
                    value={`${stats.streak.current_streak} days`}
                    color={CHART_COLORS.warning}
                />
            </section>

            {/* Main Content Grid */}
            <div className="progress__main-grid">
                {/* Score History Chart */}
                <Card variant="elevated" padding="lg" className="progress__chart-card progress__chart-card--full">
                    <div className="progress__chart-header">
                        <h2 className="progress__chart-title">Score History</h2>
                        <PeriodSelector value={period} onChange={handlePeriodChange} />
                    </div>
                    <div className="progress__chart">
                        <ResponsiveContainer width="100%" height={300}>
                            <AreaChart data={chartData}>
                                <defs>
                                    <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#14b8a6" stopOpacity={0.3} />
                                        <stop offset="95%" stopColor="#14b8a6" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="var(--border-primary)" />
                                <XAxis
                                    dataKey="date"
                                    stroke="var(--text-tertiary)"
                                    fontSize={12}
                                    tickLine={false}
                                />
                                <YAxis
                                    stroke="var(--text-tertiary)"
                                    fontSize={12}
                                    tickLine={false}
                                    domain={[0, 100]}
                                    tickFormatter={(value) => `${value}%`}
                                />
                                <Tooltip content={<CustomTooltip />} />
                                <Area
                                    type="monotone"
                                    dataKey="score"
                                    name="Score"
                                    stroke="#14b8a6"
                                    strokeWidth={2}
                                    fill="url(#scoreGradient)"
                                />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>
                </Card>

                {/* Phoneme Mastery Section */}
                <Card variant="elevated" padding="lg" className="progress__chart-card">
                    <div className="progress__chart-header">
                        <h2 className="progress__chart-title">Phoneme Mastery</h2>
                        <span className="progress__chart-subtitle">
                            {phonemeData.weak.length} needs work, {phonemeData.strong.length} mastered
                        </span>
                    </div>
                    <div className="progress__mastery-list">
                        {phonemeData.all.length > 0 ? (
                            phonemeData.all.slice(0, 8).map((p, idx) => (
                                <PhonemeMasteryBar
                                    key={idx}
                                    phoneme={p.phoneme}
                                    symbol={p.symbol}
                                    score={p.current_score || 0}
                                    attempts={p.attempts || 0}
                                />
                            ))
                        ) : (
                            <p className="progress__no-data">No phoneme data yet. Keep practicing!</p>
                        )}
                    </div>
                    {phonemeData.all.length > 8 && (
                        <button
                            className="progress__view-all-btn"
                            onClick={() => navigate('/phonemes')}
                        >
                            View All Phonemes
                            <ChevronRight size={16} />
                        </button>
                    )}
                </Card>

                {/* Session History */}
                <Card variant="elevated" padding="lg" className="progress__chart-card">
                    <div className="progress__chart-header">
                        <h2 className="progress__chart-title">Recent Sessions</h2>
                        <span className="progress__chart-subtitle">Last {period} days</span>
                    </div>
                    <div className="progress__history-list">
                        {sessionHistory.length > 0 ? (
                            sessionHistory.slice(0, 7).map((session, idx) => (
                                <SessionHistoryItem
                                    key={idx}
                                    date={session.date}
                                    score={session.score}
                                    attempts={session.attempts}
                                    duration={session.duration}
                                />
                            ))
                        ) : (
                            <p className="progress__no-data">No sessions recorded yet.</p>
                        )}
                    </div>
                </Card>

                {/* Practice Recommendations */}
                <Card variant="elevated" padding="lg" className="progress__chart-card progress__recommendations">
                    <div className="progress__chart-header">
                        <h2 className="progress__chart-title">
                            <Zap size={20} className="progress__title-icon" />
                            Focus Areas
                        </h2>
                    </div>
                    <div className="progress__recommendations-list">
                        {phonemeData.weak.length > 0 ? (
                            <>
                                <p className="progress__recommendations-intro">
                                    Based on your practice history, focus on these sounds:
                                </p>
                                <div className="progress__focus-phonemes">
                                    {phonemeData.weak.slice(0, 4).map((p, idx) => (
                                        <div key={idx} className="progress__focus-item">
                                            <span className="progress__focus-symbol">/{p.symbol || p.phoneme}/</span>
                                            <span className="progress__focus-score">
                                                {Math.round((p.current_score || 0) * 100)}%
                                            </span>
                                        </div>
                                    ))}
                                </div>
                                <button
                                    className="progress__practice-btn"
                                    onClick={() => navigate('/practice')}
                                >
                                    <Mic size={18} />
                                    Practice These Sounds
                                </button>
                            </>
                        ) : (
                            <div className="progress__mastery-complete">
                                <Award size={48} className="progress__mastery-icon" />
                                <p>Great job! You're performing well across all phonemes.</p>
                                <button
                                    className="progress__practice-btn"
                                    onClick={() => navigate('/practice')}
                                >
                                    <Mic size={18} />
                                    Continue Practice
                                </button>
                            </div>
                        )}
                    </div>
                </Card>
            </div>
        </div>
    );
}

export default Progress;
