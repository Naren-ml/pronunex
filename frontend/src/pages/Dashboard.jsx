/**
 * Dashboard Page - Three-Column Grid Layout
 * Full viewport adaptive with profile, stats, and charts
 */

import { useState, useEffect, useRef, useMemo } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import {
    Mic,
    BarChart2,
    BookOpen,
    Target,
    Flame,
    Award,
    ArrowRight,
    Sparkles,
    Trophy,
    Star,
    Zap,
    CheckCircle
} from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { useApi } from '../hooks/useApi';
import { ENDPOINTS } from '../api/endpoints';
import { Spinner } from '../components/Loader';
import { ErrorState } from '../components/ErrorState';
import './Dashboard.css';
import '../components/progress/MilestonesBadges.css'; // Import badge styles

// Progress Ring Component
function ProgressRing({ progress, size = 100, strokeWidth = 10 }) {
    const radius = (size - strokeWidth) / 2;
    const circumference = 2 * Math.PI * radius;
    const strokeDasharray = `${(progress / 100) * circumference} ${circumference}`;

    return (
        <div className="dashboard__goal-ring" style={{ width: size, height: size }}>
            <svg viewBox={`0 0 ${size} ${size}`}>
                <circle
                    className="dashboard__goal-ring-bg"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                />
                <circle
                    className="dashboard__goal-ring-progress"
                    cx={size / 2}
                    cy={size / 2}
                    r={radius}
                    strokeDasharray={strokeDasharray}
                />
            </svg>
            <span className="dashboard__goal-value">{Math.round(progress)}%</span>
        </div>
    );
}

// Sparkline Component - With gradient fill
function Sparkline({ data = [], color = '#047857' }) {
    const canvasRef = useRef(null);

    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas || data.length === 0) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = 4;

        ctx.clearRect(0, 0, width, height);

        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const range = max - min || 1;

        const points = data.map((value, index) => ({
            x: padding + (index / (data.length - 1)) * (width - padding * 2),
            y: height - padding - ((value - min) / range) * (height - padding * 2)
        }));

        // Draw gradient area fill
        const gradient = ctx.createLinearGradient(0, 0, 0, height);
        gradient.addColorStop(0, color + '30');
        gradient.addColorStop(1, color + '05');

        ctx.beginPath();
        ctx.moveTo(points[0].x, height);
        ctx.lineTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xMid = (points[i].x + points[i + 1].x) / 2;
            const yMid = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xMid, yMid);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.lineTo(points[points.length - 1].x, height);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw smooth line
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
    }, [data, color]);

    return (
        <canvas
            ref={canvasRef}
            width={120}
            height={36}
            className="dashboard__stat-sparkline"
        />
    );
}

// Weekly Progress Chart Component with Smooth Curves and Tooltips
function WeeklyChart({ data = [], labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'] }) {
    const canvasRef = useRef(null);
    const containerRef = useRef(null);
    const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, day: '', score: 0 });

    const chartData = data.length > 0 ? data : [0, 0, 0, 0, 0, 0, 0];


    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const padding = { top: 20, right: 20, bottom: 30, left: 40 };

        ctx.clearRect(0, 0, width, height);

        const chartWidth = width - padding.left - padding.right;
        const chartHeight = height - padding.top - padding.bottom;

        const max = Math.max(...chartData, 100);
        const min = 0;

        // Draw grid lines with increased opacity
        ctx.strokeStyle = 'rgba(0, 0, 0, 0.08)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding.top + (chartHeight / 4) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(width - padding.right, y);
            ctx.stroke();
        }

        // Calculate points
        const points = chartData.map((value, index) => ({
            x: padding.left + (index / (chartData.length - 1)) * chartWidth,
            y: padding.top + chartHeight - ((value - min) / (max - min)) * chartHeight
        }));

        // Draw soft gradient area fill first (behind line)
        const gradient = ctx.createLinearGradient(0, padding.top, 0, height - padding.bottom);
        gradient.addColorStop(0, 'rgba(4, 120, 87, 0.18)');
        gradient.addColorStop(0.5, 'rgba(4, 120, 87, 0.08)');
        gradient.addColorStop(1, 'rgba(4, 120, 87, 0)');

        ctx.beginPath();
        ctx.moveTo(points[0].x, height - padding.bottom);
        ctx.lineTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xMid = (points[i].x + points[i + 1].x) / 2;
            const yMid = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xMid, yMid);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.lineTo(points[points.length - 1].x, height - padding.bottom);
        ctx.fillStyle = gradient;
        ctx.fill();

        // Draw smooth bezier curve line
        ctx.beginPath();
        ctx.moveTo(points[0].x, points[0].y);
        for (let i = 0; i < points.length - 1; i++) {
            const xMid = (points[i].x + points[i + 1].x) / 2;
            const yMid = (points[i].y + points[i + 1].y) / 2;
            ctx.quadraticCurveTo(points[i].x, points[i].y, xMid, yMid);
        }
        ctx.lineTo(points[points.length - 1].x, points[points.length - 1].y);
        ctx.strokeStyle = '#047857';
        ctx.lineWidth = 2.5;
        ctx.lineCap = 'round';
        ctx.lineJoin = 'round';
        ctx.stroke();

        // Draw points
        points.forEach(p => {
            ctx.beginPath();
            ctx.arc(p.x, p.y, 4, 0, 2 * Math.PI);
            ctx.fillStyle = '#047857';
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });

        // Draw labels
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter, sans-serif';
        ctx.textAlign = 'center';
        labels.forEach((label, index) => {
            const x = padding.left + (index / (labels.length - 1)) * chartWidth;
            ctx.fillText(label, x, height - 8);
        });

        // Draw Y-axis labels
        ctx.textAlign = 'right';
        for (let i = 0; i <= 4; i++) {
            const value = Math.round(min + ((max - min) / 4) * (4 - i));
            const y = padding.top + (chartHeight / 4) * i + 4;
            ctx.fillText(value + '%', padding.left - 8, y);
        }

        // Store points for tooltip detection
        canvas.chartPoints = points;
        canvas.chartData = chartData;
        canvas.chartLabels = labels;
        canvas.chartPadding = padding;
    }, [chartData, labels]);

    const handleMouseMove = (e) => {
        const canvas = canvasRef.current;
        if (!canvas || !canvas.chartPoints) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const x = (e.clientX - rect.left) * scaleX;
        const points = canvas.chartPoints;

        // Find closest point
        let closestIdx = 0;
        let closestDist = Infinity;
        points.forEach((p, idx) => {
            const dist = Math.abs(p.x - x);
            if (dist < closestDist) {
                closestDist = dist;
                closestIdx = idx;
            }
        });

        if (closestDist < 40) {
            setTooltip({
                show: true,
                x: e.clientX - rect.left,
                y: points[closestIdx].y / scaleX - 10,
                day: canvas.chartLabels[closestIdx],
                score: canvas.chartData[closestIdx]
            });
        } else {
            setTooltip({ ...tooltip, show: false });
        }
    };

    const handleMouseLeave = () => {
        setTooltip({ ...tooltip, show: false });
    };

    return (
        <div
            ref={containerRef}
            className="dashboard__chart-container"
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ position: 'relative' }}
        >
            <canvas
                ref={canvasRef}
                width={600}
                height={180}
                className="dashboard__chart-canvas"
            />
            {tooltip.show && (
                <div
                    className="dashboard__chart-tooltip"
                    style={{
                        position: 'absolute',
                        left: tooltip.x,
                        top: tooltip.y,
                        transform: 'translate(-50%, -100%)',
                        background: '#0f172a',
                        color: '#fff',
                        padding: '6px 10px',
                        borderRadius: '6px',
                        fontSize: '12px',
                        fontWeight: '500',
                        whiteSpace: 'nowrap',
                        pointerEvents: 'none',
                        zIndex: 10
                    }}
                >
                    {tooltip.day}: {tooltip.score}%
                </div>
            )}
        </div>
    );
}

// Milestone definitions with icons and thresholds
const MILESTONE_DEFINITIONS = [
    {
        id: 'first_practice',
        name: 'First Steps',
        description: 'Complete your first practice session',
        icon: Star,
        threshold: 1,
        field: 'total_attempts',
        color: '#f59e0b'
    },
    {
        id: 'streak_7',
        name: 'Week Warrior',
        description: 'Achieve a 7-day practice streak',
        icon: Flame,
        threshold: 7,
        field: 'current_streak',
        color: '#ef4444'
    },
    {
        id: 'attempts_50',
        name: 'Dedicated Learner',
        description: 'Complete 50 practice attempts',
        icon: Target,
        threshold: 50,
        field: 'total_attempts',
        color: '#10b981'
    },
    {
        id: 'attempts_100',
        name: 'Century Club',
        description: 'Complete 100 practice attempts',
        icon: Trophy,
        threshold: 100,
        field: 'total_attempts',
        color: '#14b8a6'
    },
];

function MilestoneBadge({ milestone, progress, isEarned, isNew }) {
    const Icon = milestone.icon;
    const progressPercent = Math.min((progress / milestone.threshold) * 100, 100);

    return (
        <div
            className={`milestone-badge ${isEarned ? 'milestone-badge--earned' : ''} ${isNew ? 'milestone-badge--new' : ''}`}
            title={milestone.description}
        >
            <div
                className="milestone-badge__icon-wrapper"
                style={{
                    '--milestone-color': milestone.color,
                    '--progress-percent': `${progressPercent}%`
                }}
            >
                <Icon
                    size={24}
                    className="milestone-badge__icon"
                    aria-hidden="true"
                />
                {isEarned && (
                    <CheckCircle
                        size={12}
                        className="milestone-badge__check"
                        aria-hidden="true"
                    />
                )}
                {!isEarned && (
                    <svg className="milestone-badge__progress-ring" viewBox="0 0 36 36">
                        <path
                            className="milestone-badge__progress-bg"
                            d="M18 2.0845
                               a 15.9155 15.9155 0 0 1 0 31.831
                               a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                        <path
                            className="milestone-badge__progress-fill"
                            strokeDasharray={`${progressPercent}, 100`}
                            d="M18 2.0845
                               a 15.9155 15.9155 0 0 1 0 31.831
                               a 15.9155 15.9155 0 0 1 0 -31.831"
                        />
                    </svg>
                )}
            </div>
            <span className="milestone-badge__name">{milestone.name}</span>
            {!isEarned && (
                <span className="milestone-badge__progress-text">
                    {Math.round(progress)}/{milestone.threshold}
                </span>
            )}
        </div>
    );
}




export function Dashboard() {
    const { user } = useAuth();
    const navigate = useNavigate();
    const { data: progress, isLoading, error, refetch } = useApi(ENDPOINTS.ANALYTICS.PROGRESS);

    // Calculate milestones directly from progress data
    const milestones = useMemo(() => {
        const stats = progress || {};
        return MILESTONE_DEFINITIONS.map(milestone => {
            let currentValue = 0;

            switch (milestone.field) {
                case 'total_attempts':
                    currentValue = stats.attempt_stats?.total_attempts || stats.total_attempts || 0;
                    break;
                case 'current_streak':
                    currentValue = stats.streak?.current_streak || 0;
                    break;
                case 'overall_average_score':
                    currentValue = stats.attempt_stats?.avg_score || stats.overall_average_score || 0;
                    break;
                case 'mastered_phonemes_count':
                    currentValue = stats.current_strong_phonemes?.length || 0;
                    break;
                default:
                    currentValue = 0;
            }

            const isEarned = currentValue >= milestone.threshold;

            return {
                ...milestone,
                progress: currentValue,
                isEarned,
                isNew: false
            };
        });
    }, [progress]);

    if (isLoading) {
        return (
            <div className="dashboard-loading">
                <Spinner size="lg" />
                <p>Loading your progress...</p>
            </div>
        );
    }

    if (error) {
        return (
            <div className="dashboard-error">
                <ErrorState
                    icon="server"
                    title="Failed to load dashboard"
                    message="We could not load your progress data. Please try again."
                    onRetry={refetch}
                />
            </div>
        );
    }

    const stats = progress || {};

    // Normalize backend response
    const normalizedStats = {
        total_attempts: stats.total_attempts || 0,
        average_score: stats.overall_average_score ?? stats.average_score ?? 0,
        streak_days: stats.streak?.current_streak ?? stats.streak_days ?? 0,
        weak_phonemes: stats.current_weak_phonemes || [],
        weak_phonemes_count: stats.current_weak_phonemes?.length ?? stats.weak_phonemes_count ?? 0,
        daily_goal_progress: stats.daily_goal_progress || 0,
        weekly_scores: stats.weekly_scores || [0, 0, 0, 0, 0, 0, 0],
        weekly_labels: stats.weekly_labels || ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    };

    const hasNoActivity = normalizedStats.total_attempts === 0;
    const userInitials = (user?.full_name || user?.username || 'U').charAt(0).toUpperCase();
    const userName = user?.full_name || user?.username || 'User';
    const userLevel = normalizedStats.average_score >= 0.8 ? 'Advanced' :
        normalizedStats.average_score >= 0.5 ? 'Intermediate' : 'Beginner';

    // Mock weak phonemes for display
    const weakPhonemes = normalizedStats.weak_phonemes.length > 0
        ? normalizedStats.weak_phonemes.slice(0, 3)
        : [
            { phoneme: 'th', example: 'think', score: 45 },
            { phoneme: 'r', example: 'red', score: 52 },
            { phoneme: 'l', example: 'love', score: 58 },
        ];

    return (
        <div className="dashboard">
            {/* Header */}
            <header className="dashboard__header">
                <div className="dashboard__welcome">
                    <h1 className="dashboard__title">
                        Welcome back, {userName}!
                    </h1>
                    <p className="dashboard__subtitle">
                        {hasNoActivity
                            ? 'Start your first practice session to begin improving.'
                            : 'Continue your journey to better pronunciation.'}
                    </p>
                </div>
                <Link to="/practice" className="dashboard__cta btn btn--primary btn--lg">
                    <Mic size={20} />
                    <span>Start Practice</span>
                </Link>
            </header>

            {/* Main Grid */}
            <div className="dashboard__grid">
                {/* Left Column - Profile & Goals */}
                <aside className="dashboard__left-column">
                    {/* Profile Card */}
                    <div className="dashboard__profile-card">
                        <div className="dashboard__avatar">{userInitials}</div>
                        <div className="dashboard__user-info">
                            <h2 className="dashboard__user-name">{userName}</h2>
                            <span className="dashboard__user-level">{userLevel}</span>
                        </div>

                        {/* Daily Goal */}
                        <div className="dashboard__daily-goal">
                            <ProgressRing progress={normalizedStats.daily_goal_progress} />
                            <span className="dashboard__goal-label">Daily Goal Progress</span>
                        </div>
                    </div>

                    {/* Achievements */}
                    <div className="dashboard__achievements">
                        <h3 className="dashboard__section-title">Recent Badges</h3>
                        <div className="dashboard__badges" style={{
                            display: 'grid',
                            gridTemplateColumns: 'repeat(auto-fill, minmax(80px, 1fr))',
                            gap: '1rem',
                            marginTop: '0.5rem'
                        }}>
                            {milestones.map(milestone => (
                                <MilestoneBadge
                                    key={milestone.id}
                                    milestone={milestone}
                                    progress={milestone.progress}
                                    isEarned={milestone.isEarned}
                                />
                            ))}
                        </div>
                    </div>
                </aside>

                {/* Center Column - Stats & Charts */}
                <main className="dashboard__center-column">
                    {/* Stats Grid */}
                    <div className="dashboard__stats-grid">
                        <div className="dashboard__stat-card">
                            <div className="dashboard__stat-header">
                                <span className="dashboard__stat-label">Total Attempts</span>
                                <div className="dashboard__stat-icon">
                                    <Target size={18} />
                                </div>
                            </div>
                            <span className="dashboard__stat-value">{normalizedStats.total_attempts}</span>
                            <Sparkline data={[5, 8, 12, 15, 18, 22, 25]} color="#10b981" />
                        </div>

                        <div className="dashboard__stat-card">
                            <div className="dashboard__stat-header">
                                <span className="dashboard__stat-label">Average Score</span>
                                <div className="dashboard__stat-icon">
                                    <Award size={18} />
                                </div>
                            </div>
                            <span className="dashboard__stat-value">
                                {Math.round(normalizedStats.average_score * 100)}%
                            </span>
                            <Sparkline data={normalizedStats.weekly_scores} color="#10b981" />
                        </div>

                        <div className="dashboard__stat-card">
                            <div className="dashboard__stat-header">
                                <span className="dashboard__stat-label">Day Streak</span>
                                <div className="dashboard__stat-icon" style={{ background: '#fef3c7', color: '#d97706' }}>
                                    <Flame size={18} />
                                </div>
                            </div>
                            <span className="dashboard__stat-value">{normalizedStats.streak_days}</span>
                        </div>

                        <div className="dashboard__stat-card">
                            <div className="dashboard__stat-header">
                                <span className="dashboard__stat-label">Focus Areas</span>
                                <div className="dashboard__stat-icon" style={{ background: '#dbeafe', color: '#2563eb' }}>
                                    <BookOpen size={18} />
                                </div>
                            </div>
                            <span className="dashboard__stat-value">{normalizedStats.weak_phonemes_count}</span>
                        </div>
                    </div>

                    {/* Weekly Progress Chart */}
                    <div className="dashboard__chart-card">
                        <div className="dashboard__chart-header">
                            <h3 className="dashboard__chart-title">Weekly Progress</h3>
                            <span className="dashboard__chart-period">Last 7 days</span>
                        </div>
                        <WeeklyChart data={normalizedStats.weekly_scores} labels={normalizedStats.weekly_labels} />
                    </div>

                    {/* AI Insight */}
                    <div className="dashboard__ai-insight">
                        <div className="dashboard__ai-icon">
                            <Sparkles size={22} />
                        </div>
                        <div className="dashboard__ai-content">
                            <h4 className="dashboard__ai-label">AI Insight</h4>
                            <p className="dashboard__ai-text">
                                You're doing great with vowels! Try focusing on the "th" and "r" sounds today.
                                Practice with tongue twisters for faster improvement.
                            </p>
                        </div>
                    </div>
                </main>

                {/* Right Column - Sidebar */}
                <aside className="dashboard__right-column">
                    {/* Quick Actions */}
                    <div className="dashboard__quick-actions">
                        <h3 className="dashboard__section-title">Quick Actions</h3>
                        <div className="dashboard__action-list">
                            <Link to="/practice" className="dashboard__action-item">
                                <div className="dashboard__action-icon dashboard__action-icon--primary">
                                    <Mic size={20} />
                                </div>
                                <div className="dashboard__action-text">
                                    <h4>Practice Session</h4>
                                    <p>Record and assess</p>
                                </div>
                                <ArrowRight size={16} className="dashboard__action-arrow" />
                            </Link>

                            <Link to="/progress" className="dashboard__action-item">
                                <div className="dashboard__action-icon dashboard__action-icon--success">
                                    <BarChart2 size={20} />
                                </div>
                                <div className="dashboard__action-text">
                                    <h4>View Progress</h4>
                                    <p>Track improvement</p>
                                </div>
                                <ArrowRight size={16} className="dashboard__action-arrow" />
                            </Link>

                            <Link to="/phonemes" className="dashboard__action-item">
                                <div className="dashboard__action-icon dashboard__action-icon--info">
                                    <BookOpen size={20} />
                                </div>
                                <div className="dashboard__action-text">
                                    <h4>Phoneme Library</h4>
                                    <p>Learn sounds</p>
                                </div>
                                <ArrowRight size={16} className="dashboard__action-arrow" />
                            </Link>
                        </div>
                    </div>

                    {/* Words to Review */}
                    <div className="dashboard__words-review">
                        <h3 className="dashboard__section-title">Focus Phonemes</h3>
                        <div className="dashboard__word-list">
                            {weakPhonemes.map((item, idx) => (
                                <div key={idx} className="dashboard__word-item">
                                    <div className="dashboard__word-info">
                                        <span className="dashboard__word-phoneme">
                                            /{typeof item === 'string' ? item : item.phoneme}/
                                        </span>
                                        <span className="dashboard__word-example">
                                            {typeof item === 'string' ? '' : item.example}
                                        </span>
                                    </div>
                                    <button
                                        className="dashboard__word-practice"
                                        onClick={() => navigate('/practice')}
                                    >
                                        Practice
                                    </button>
                                </div>
                            ))}
                        </div>
                    </div>
                </aside>


                {/* Bottom Section - 2x2 Grid */}
                <section className="dashboard__bottom-section" style={{
                    gridColumn: '1 / -1',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(2, 1fr)',
                    gap: '1.5rem',
                    marginTop: '0.5rem'
                }}>
                    <SessionHistory />
                    <QuickPractice />
                    <PhonemeMastery />
                    <LearningTips />
                </section>

            </div>
        </div>
    );
}

// Lazy load bottom components to avoid circular deps if any
import { SessionHistory } from '../components/dashboard/bottom/SessionHistory';
import { QuickPractice } from '../components/dashboard/bottom/QuickPractice';
import { PhonemeMastery } from '../components/dashboard/bottom/PhonemeMastery';
import { LearningTips } from '../components/dashboard/bottom/LearningTips';

export default Dashboard;
