import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ArrowRight } from 'lucide-react';
import './Navbar.css';

export default function Navbar() {
    const [isScrolled, setIsScrolled] = useState(false);
    const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

    const navLinks = [
        { name: 'About Us', href: '/about' },
        { name: 'Pricing', href: '/pricing' },
        { name: 'Docs', href: '/docs' },
        { name: 'Features', href: '/features' }, // <--- Fixed this to point to the new page
        { name: 'How It Works', href: '/#how-it-works' },
        { name: 'Practice', href: '/practice' },
    ];

    useEffect(() => {
        const handleScroll = () => setIsScrolled(window.scrollY > 20);
        window.addEventListener('scroll', handleScroll);
        return () => window.removeEventListener('scroll', handleScroll);
    }, []);

    return (
        <nav className={`navbar-enterprise ${isScrolled ? 'scrolled' : ''}`}>
            <div className="navbar-container-enterprise">
                {/* 1. Logo */}
                <Link to="/" className="navbar-logo-enterprise">
                    <img src="/icon.png" alt="Pronunex" className="navbar-logo-img" />
                    <span className="navbar-logo-text">Pronunex</span>
                </Link>

                {/* 2. CENTER LINKS */}
                <div className="navbar-desktop-links" style={{ display: 'flex', gap: '2rem' }}> 
                    {navLinks.map((link) => (
                        <Link 
                            key={link.name} 
                            to={link.href} 
                            className="navbar-link-enterprise"
                            style={{ color: '#475569', fontSize: '1rem', fontWeight: '500', textDecoration: 'none' }}
                        >
                            {link.name}
                        </Link>
                    ))}
                </div>

                {/* 3. Right Side Buttons */}
                <div className="navbar-actions-enterprise">
                    <Link to="/login" className="navbar-btn-secondary">Sign In</Link>
                    <Link to="/signup" className="navbar-btn-primary">
                        <span>Get Started</span>
                        <ArrowRight size={16} />
                    </Link>
                </div>

                {/* Mobile Toggle */}
                <button 
                    className="navbar-mobile-toggle"
                    onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
                >
                    {isMobileMenuOpen ? <X /> : <Menu />}
                </button>
            </div>

            {/* Mobile Menu */}
            <AnimatePresence>
                {isMobileMenuOpen && (
                    <motion.div 
                        className="navbar-mobile-menu"
                        initial={{ opacity: 0, y: -20 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0, y: -20 }}
                    >
                        {navLinks.map((link) => (
                            <Link key={link.name} to={link.href} onClick={() => setIsMobileMenuOpen(false)}>
                                {link.name}
                            </Link>
                        ))}
                    </motion.div>
                )}
            </AnimatePresence>
        </nav>
    );
}