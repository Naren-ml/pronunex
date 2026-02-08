/**
 * How It Works Page - Dedicated Route
 * Premium enterprise-grade page showing the 3-step process
 */

import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Mic, Brain, Target, ArrowRight, CheckCircle, Zap, Shield, BarChart3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import './HowItWorks.css';

const steps = [
    {
        number: '01',
        icon: Mic,
        title: 'Record Your Voice',
        shortTitle: 'Record',
        description: 'Read aloud curated sentences designed for pronunciation practice. Our system captures your audio with high-fidelity accuracy for precise phoneme-level analysis.',
        features: [
            'High-quality audio capture',
            'Noise reduction technology',
            'Works on any device'
        ],
        color: '#10b981'
    },
    {
        number: '02',
        icon: Brain,
        title: 'AI-Powered Analysis',
        shortTitle: 'Analyze',
        description: 'Our advanced AI uses forced alignment and pronunciation scoring algorithms to identify exactly where you struggle—down to individual phonemes and sounds.',
        features: [
            'Phoneme-level detection',
            'Real-time processing',
            'Detailed accuracy scores'
        ],
        color: '#8b5cf6'
    },
    {
        number: '03',
        icon: Target,
        title: 'Personalized Roadmap',
        shortTitle: 'Improve',
        description: 'Receive a customized learning path focusing on your specific problem areas. Practice exercises adapt to your progress for continuous improvement.',
        features: [
            'Custom exercise plans',
            'Adaptive difficulty',
            'Track your progress'
        ],
        color: '#f59e0b'
    },
];

const benefits = [
    {
        icon: Zap,
        title: 'Instant Feedback',
        description: 'Get results in milliseconds, not minutes'
    },
    {
        icon: Shield,
        title: 'Privacy First',
        description: 'Your voice data is processed securely'
    },
    {
        icon: BarChart3,
        title: 'Track Progress',
        description: 'Visual charts show your improvement'
    },
];

const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
        opacity: 1,
        transition: {
            staggerChildren: 0.15,
            delayChildren: 0.1,
        },
    },
};

const itemVariants = {
    hidden: { opacity: 0, y: 40 },
    visible: {
        opacity: 1,
        y: 0,
        transition: { duration: 0.6, ease: [0.4, 0, 0.2, 1] },
    },
};

export default function HowItWorks() {
    const heroRef = useRef(null);
    const stepsRef = useRef(null);
    const benefitsRef = useRef(null);

    const heroInView = useInView(heroRef, { once: true, margin: '-50px' });
    const stepsInView = useInView(stepsRef, { once: true, margin: '-100px' });
    const benefitsInView = useInView(benefitsRef, { once: true, margin: '-100px' });

    return (
        <div className="how-it-works-page">
            {/* Hero Section */}
            <section className="hiw-hero" ref={heroRef}>
                <motion.div
                    className="hiw-hero-content"
                    initial={{ opacity: 0, y: 30 }}
                    animate={heroInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.6 }}
                >
                    <span className="hiw-hero-badge">Simple 3-Step Process</span>
                    <h1 className="hiw-hero-title">
                        How <span className="hiw-gradient-text">Pronunex</span> Works
                    </h1>
                    <p className="hiw-hero-subtitle">
                        Master your pronunciation with our AI-powered platform.
                        Record, analyze, and improve—it's that simple.
                    </p>
                    <div className="hiw-hero-cta">
                        <Link to="/signup" className="hiw-btn-primary">
                            <span>Start Free Trial</span>
                            <ArrowRight size={18} />
                        </Link>
                        <Link to="/features" className="hiw-btn-secondary">
                            See All Features
                        </Link>
                    </div>
                </motion.div>

                {/* Floating Elements */}
                <div className="hiw-hero-decoration">
                    <div className="hiw-float-circle hiw-float-1"></div>
                    <div className="hiw-float-circle hiw-float-2"></div>
                    <div className="hiw-float-circle hiw-float-3"></div>
                </div>
            </section>

            {/* Steps Section */}
            <section className="hiw-steps-section" ref={stepsRef}>
                <div className="hiw-container">
                    <motion.div
                        className="hiw-steps-grid"
                        variants={containerVariants}
                        initial="hidden"
                        animate={stepsInView ? "visible" : "hidden"}
                    >
                        {steps.map((step, index) => (
                            <motion.div
                                key={step.number}
                                className="hiw-step-card"
                                variants={itemVariants}
                            >
                                {/* Step Number Badge */}
                                <div
                                    className="hiw-step-number"
                                    style={{ background: `linear-gradient(135deg, ${step.color} 0%, ${step.color}dd 100%)` }}
                                >
                                    {step.number}
                                </div>

                                {/* Icon */}
                                <div
                                    className="hiw-step-icon"
                                    style={{ background: `${step.color}15`, color: step.color }}
                                >
                                    <step.icon size={32} strokeWidth={1.5} />
                                </div>

                                {/* Content */}
                                <h3 className="hiw-step-title">{step.title}</h3>
                                <p className="hiw-step-description">{step.description}</p>

                                {/* Features List */}
                                <ul className="hiw-step-features">
                                    {step.features.map((feature, i) => (
                                        <li key={i}>
                                            <CheckCircle size={16} style={{ color: step.color }} />
                                            <span>{feature}</span>
                                        </li>
                                    ))}
                                </ul>

                                {/* Connector Line */}
                                {index < steps.length - 1 && (
                                    <div className="hiw-step-connector">
                                        <ArrowRight size={20} />
                                    </div>
                                )}
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* Benefits Section */}
            <section className="hiw-benefits-section" ref={benefitsRef}>
                <div className="hiw-container">
                    <motion.h2
                        className="hiw-benefits-title"
                        initial={{ opacity: 0, y: 20 }}
                        animate={benefitsInView ? { opacity: 1, y: 0 } : {}}
                        transition={{ duration: 0.5 }}
                    >
                        Why Our Approach Works
                    </motion.h2>

                    <motion.div
                        className="hiw-benefits-grid"
                        variants={containerVariants}
                        initial="hidden"
                        animate={benefitsInView ? "visible" : "hidden"}
                    >
                        {benefits.map((benefit, index) => (
                            <motion.div
                                key={index}
                                className="hiw-benefit-card"
                                variants={itemVariants}
                            >
                                <div className="hiw-benefit-icon">
                                    <benefit.icon size={24} />
                                </div>
                                <h4 className="hiw-benefit-title">{benefit.title}</h4>
                                <p className="hiw-benefit-text">{benefit.description}</p>
                            </motion.div>
                        ))}
                    </motion.div>
                </div>
            </section>

            {/* CTA Section */}
            <section className="hiw-cta-section">
                <div className="hiw-container">
                    <div className="hiw-cta-card">
                        <h2 className="hiw-cta-title">Ready to Improve Your Pronunciation?</h2>
                        <p className="hiw-cta-text">
                            Join thousands of learners who have transformed their speaking skills with Pronunex.
                        </p>
                        <Link to="/signup" className="hiw-cta-button">
                            <span>Get Started for Free</span>
                            <ArrowRight size={18} />
                        </Link>
                    </div>
                </div>
            </section>
        </div>
    );
}
