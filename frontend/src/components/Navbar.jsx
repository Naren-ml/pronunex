/**
 * Navbar Component - Professional EdTech Layout
 * Full-width with streak counter, notifications, and enhanced active states
 */

import { NavLink, useNavigate, useLocation } from 'react-router-dom';
import {
    Home,
    Mic,
    BarChart2,
    BookOpen,
    LogOut,
    User,
    Menu,
    X,
    Shield,
    Flame,
    Bell,
    ChevronDown,
    Settings
} from 'lucide-react';
import { useState, useEffect, useRef, useMemo } from 'react';
import { useAuth } from '../context/AuthContext';
import { useApi } from '../hooks/useApi';
import { ENDPOINTS } from '../api/endpoints';
import './Navbar.css';

const NAV_ITEMS = [
    { path: '/dashboard', label: 'Dashboard', icon: Home },
    { path: '/practice', label: 'Practice', icon: Mic },
    { path: '/progress', label: 'Progress', icon: BarChart2 },
    { path: '/phonemes', label: 'Phonemes', icon: BookOpen },
];

export function Navbar() {
    const { user, isAuthenticated, logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
    const dropdownRef = useRef(null);

    // Fetch user progress for streak - refetch when location changes
    const { data: progress, refetch: refetchProgress } = useApi(
        isAuthenticated ? ENDPOINTS.ANALYTICS.PROGRESS : null,
        { deps: [location.pathname] }
    );

    // Extract streak - handle multiple possible structures from API
    const streakDays = useMemo(() => {
        if (!progress) return 0;

        // Try progress.streak.current_streak (serialized StreakRecord)
        if (progress.streak?.current_streak !== undefined) {
            return progress.streak.current_streak;
        }

        // Try progress.streak directly if it's a number
        if (typeof progress.streak === 'number') {
            return progress.streak;
        }

        // Fallback to streak_days field
        if (progress.streak_days !== undefined) {
            return progress.streak_days;
        }

        return 0;
    }, [progress]);

    // Close dropdown when clicking outside
    useEffect(() => {
        const handleClickOutside = (event) => {
            if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
                setIsUserMenuOpen(false);
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, []);

    const handleLogout = async () => {
        await logout();
        navigate('/login');
    };

    const toggleMobileMenu = () => {
        setIsMobileMenuOpen(!isMobileMenuOpen);
    };

    const closeMobileMenu = () => {
        setIsMobileMenuOpen(false);
    };

    const userName = user?.full_name || user?.username || 'User';
    const userInitial = userName.charAt(0).toUpperCase();
    const displayName = userName.split(' ')[0]; // First name only

    return (
        <nav className="navbar" role="navigation" aria-label="Main navigation">
            <div className="navbar__container">
                {/* Left Section - Branding */}
                <div className="navbar__brand">
                    <NavLink to="/" className="navbar__logo" onClick={closeMobileMenu}>
                        <img
                            src="/icon.png"
                            alt="Pronunex"
                            className="navbar__logo-icon"
                        />
                        <span className="navbar__logo-text">Pronunex</span>
                    </NavLink>
                </div>

                {/* Center Section - Desktop Navigation */}
                {isAuthenticated && (
                    <ul className="navbar__nav">
                        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
                            <li key={path} className="navbar__nav-item">
                                <NavLink
                                    to={path}
                                    className={({ isActive }) =>
                                        `navbar__nav-link ${isActive ? 'navbar__nav-link--active' : ''}`
                                    }
                                >
                                    <Icon size={18} />
                                    <span>{label}</span>
                                </NavLink>
                            </li>
                        ))}
                    </ul>
                )}

                {/* Right Section - Utilities & Profile */}
                <div className="navbar__actions">
                    {isAuthenticated ? (
                        <>
                            {/* Streak Counter */}
                            <div className={`navbar__streak ${streakDays > 0 ? 'navbar__streak--active' : ''}`} title={`${streakDays} day streak`}>
                                <Flame size={16} className="navbar__streak-icon" />
                                <span className="navbar__streak-value">{streakDays}</span>
                            </div>

                            {/* Notifications */}
                            <div className="navbar__notifications">
                                <button
                                    type="button"
                                    className="navbar__notification-btn"
                                    aria-label="Notifications"
                                    title="Notifications"
                                >
                                    <Bell size={20} />
                                    {/* Badge for unread notifications */}
                                    <span className="navbar__notification-badge" />
                                </button>
                            </div>

                            {/* User Menu */}
                            <div className="navbar__user-menu" ref={dropdownRef}>
                                <button
                                    type="button"
                                    className="navbar__user-button"
                                    onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
                                    aria-expanded={isUserMenuOpen}
                                    aria-haspopup="true"
                                >
                                    <div className="navbar__user-avatar">{userInitial}</div>
                                    <span className="navbar__user-name">{displayName}</span>
                                    <ChevronDown size={16} className="navbar__user-chevron" />
                                </button>

                                {isUserMenuOpen && (
                                    <div className="navbar__dropdown">
                                        <div className="navbar__dropdown-header">
                                            <p className="navbar__dropdown-email">
                                                {user?.email || 'user@example.com'}
                                            </p>
                                        </div>

                                        <NavLink
                                            to="/profile"
                                            className="navbar__dropdown-item"
                                            onClick={() => setIsUserMenuOpen(false)}
                                        >
                                            <User size={16} />
                                            <span>My Profile</span>
                                        </NavLink>

                                        <NavLink
                                            to="/settings"
                                            className="navbar__dropdown-item"
                                            onClick={() => setIsUserMenuOpen(false)}
                                        >
                                            <Settings size={16} />
                                            <span>Settings</span>
                                        </NavLink>

                                        {user?.is_staff && (
                                            <NavLink
                                                to="/admin"
                                                className="navbar__dropdown-item"
                                                onClick={() => setIsUserMenuOpen(false)}
                                            >
                                                <Shield size={16} />
                                                <span>Admin Panel</span>
                                            </NavLink>
                                        )}

                                        <div className="navbar__dropdown-divider" />

                                        <button
                                            type="button"
                                            className="navbar__dropdown-item navbar__dropdown-item--danger"
                                            onClick={() => {
                                                setIsUserMenuOpen(false);
                                                handleLogout();
                                            }}
                                        >
                                            <LogOut size={16} />
                                            <span>Logout</span>
                                        </button>
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <NavLink to="/login" className="btn btn--primary btn--sm">
                            Login
                        </NavLink>
                    )}

                    {/* Mobile menu toggle */}
                    <button
                        type="button"
                        className="navbar__mobile-toggle"
                        onClick={toggleMobileMenu}
                        aria-expanded={isMobileMenuOpen}
                        aria-label={isMobileMenuOpen ? 'Close menu' : 'Open menu'}
                    >
                        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </button>
                </div>
            </div>

            {/* Mobile Navigation */}
            {isAuthenticated && isMobileMenuOpen && (
                <div className="navbar__mobile-menu">
                    <ul className="navbar__mobile-nav">
                        {NAV_ITEMS.map(({ path, label, icon: Icon }) => (
                            <li key={path}>
                                <NavLink
                                    to={path}
                                    className={({ isActive }) =>
                                        `navbar__mobile-link ${isActive ? 'navbar__mobile-link--active' : ''}`
                                    }
                                    onClick={closeMobileMenu}
                                >
                                    <Icon size={20} />
                                    <span>{label}</span>
                                </NavLink>
                            </li>
                        ))}
                        <li>
                            <NavLink
                                to="/profile"
                                className="navbar__mobile-link"
                                onClick={closeMobileMenu}
                            >
                                <User size={20} />
                                <span>My Profile</span>
                            </NavLink>
                        </li>
                        <li>
                            <button
                                type="button"
                                className="navbar__mobile-link navbar__mobile-logout"
                                onClick={() => {
                                    handleLogout();
                                    closeMobileMenu();
                                }}
                            >
                                <LogOut size={20} />
                                <span>Logout</span>
                            </button>
                        </li>
                    </ul>
                </div>
            )}
        </nav>
    );
}

export default Navbar;
