import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ArrowRight, Info, DollarSign, FileText, Sparkles, Cog, Mic, LayoutDashboard } from 'lucide-react';
import { useAuth } from '../../context/AuthContext';
import './Navbar.css';

export default function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
    const [hoveredLink, setHoveredLink] = useState(null);
    const location = useLocation();
    const { isAuthenticated } = useAuth();

    const navLinks = [
        { name: 'About Us', href: '/about', icon: Info },
        { name: 'Pricing', href: '/pricing', icon: DollarSign },
        { name: 'Docs', href: '/docs', icon: FileText },
        { name: 'Features', href: '/features', icon: Sparkles },
        { name: 'How It Works', href: '/how-it-works', icon: Cog },
        { name: 'Practice', href: '/practice', icon: Mic },
    ];

    useEffect(() => {
        const handleScroll = () => setIsScrolled(window.scrollY > 20);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    const isActive = (href) => {
        if (href === '/') return location.pathname === '/';
        return location.pathname.startsWith(href.split('#')[0]);
    };

    // Handle click for hash links with smooth scroll
    const handleNavClick = (e, href) => {
        if (href.includes('#')) {
            const [path, hash] = href.split('#');
            const targetId = hash;

            // If we're already on the landing page, just scroll
            if (location.pathname === '/' || path === '/') {
                e.preventDefault();
                const element = document.getElementById(targetId);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }
            // If navigating from another page, the LandingPage will handle the scroll
        }
    };

    return (
        <nav className={`navbar-enterprise ${isScrolled ? 'scrolled' : ''}`}>
            <div className="navbar-container-enterprise">
                {/* LEFT: Branding */}
                <Link to="/" className="navbar-logo-enterprise">
                    <div className="navbar-logo-wrapper">
                        <img src="/icon.png" alt="Pronunex" className="navbar-logo-img" />
                    </div>
                    <span className="navbar-logo-text">Pronunex</span>
                </Link>

                {/* CENTER: Dynamic Navigation - Floating Pill */}
                <div className="navbar-center-pill">
                    <div className="navbar-pill-container">
                        {navLinks.map((link) => {
                            const Icon = link.icon;
                            const active = isActive(link.href);
                            const hovered = hoveredLink === link.name;

                            return (
                                <Link
                                    key={link.name}
                                    to={link.href}
                                    className={`navbar-pill-link ${active ? 'active' : ''}`}
                                    onMouseEnter={() => setHoveredLink(link.name)}
                                    onMouseLeave={() => setHoveredLink(null)}
                                    onClick={(e) => handleNavClick(e, link.href)}
                                >
                                    <span className="navbar-pill-icon-wrapper">
                                        <Icon
                                            size={16}
                                            className={`navbar-pill-icon ${active || hovered ? 'filled' : ''}`}
                                            strokeWidth={active || hovered ? 2.5 : 1.5}
                                        />
                                    </span>
                                    <span className="navbar-pill-text">{link.name}</span>
                                    <span className="navbar-active-underline"></span>
                                </Link>
                            );
                        })}
                    </div>
                </div>

                {/* RIGHT: Personalization & Utilities */}
                <div className="navbar-actions-enterprise">
                    {isAuthenticated ? (
                        <Link to="/dashboard" className="navbar-btn-primary">
                            <LayoutDashboard size={16} className="navbar-btn-icon" />
                            <span>Dashboard</span>
                        </Link>
                    ) : (
                        <>
                            <Link to="/login" className="navbar-btn-secondary">
                                <span>Sign In</span>
                            </Link>
                            <Link to="/signup" className="navbar-btn-primary">
                                <span>Get Started</span>
                                <ArrowRight size={16} className="navbar-btn-arrow" />
                            </Link>
                        </>
                    )}
                </div>

                {/* Mobile Toggle */}
                <button
                    className="navbar-mobile-toggle"
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                    aria-label="Toggle menu"
                >
                    <motion.div
                        initial={false}
                        animate={{ rotate: isMobileMenuOpen ? 90 : 0 }}
                        transition={{ duration: 0.2 }}
                    >
                        {isMobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
                    </motion.div>
                </button>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div
                        className="navbar-mobile-menu"
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: 'auto' }}
                        exit={{ opacity: 0, height: 0 }}
                        transition={{ duration: 0.3, ease: 'easeInOut' }}
                    >
                        <div className="navbar-mobile-links">
                            {navLinks.map((link, index) => {
                                const Icon = link.icon;
                                const active = isActive(link.href);

                                return (
                                    <motion.div
                                        key={link.name}
                                        initial={{ opacity: 0, x: -20 }}
                                        animate={{ opacity: 1, x: 0 }}
                                        transition={{ delay: index * 0.05 }}
                                    >
                                        <Link
                                            to={link.href}
                                            className={`navbar-mobile-link ${active ? 'active' : ''}`}
                                            onClick={(e) => {
                                                handleNavClick(e, link.href);
                                                setIsMobileMenuOpen(false);
                                            }}
                                        >
                                            <Icon size={18} className="navbar-mobile-icon" />
                                            <span>{link.name}</span>
                                            {active && <div className="navbar-mobile-active-dot" />}
                                        </Link>
                                    </motion.div>
                                );
                            })}
                        </div>
                        <div className="navbar-mobile-actions">
                            {isAuthenticated ? (
                                <Link
                                    to="/dashboard"
                                    className="navbar-mobile-btn-primary"
                                    onClick={() => setIsMobileMenuOpen(false)}
                                >
                                    <LayoutDashboard size={16} />
                                    <span>Go to Dashboard</span>
                                </Link>
                            ) : (
                                <>
                                    <Link
                                        to="/login"
                                        className="navbar-mobile-btn-secondary"
                                        onClick={() => setIsMobileMenuOpen(false)}
                                    >
                                        Sign In
                                    </Link>
                                    <Link
                                        to="/signup"
                                        className="navbar-mobile-btn-primary"
                                        onClick={() => setIsMobileMenuOpen(false)}
                                    >
                                        <span>Get Started</span>
                                        <ArrowRight size={16} />
                                    </Link>
                                </>
                            )}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
}