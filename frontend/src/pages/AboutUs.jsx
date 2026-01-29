import React from 'react';
import { motion } from 'framer-motion';
import { Users, Target, Heart, Globe, Award, Sparkles } from 'lucide-react';
// ensure this path is correct based on our previous fix!
import '../components/landing/HeroSection.css'; 

export default function AboutUs() {
    const stats = [
        { label: "Active Learners", value: "10k+" },
        { label: "Phonemes Tracked", value: "44" },
        { label: "Speech Accuracy", value: "95%" },
    ];

    const values = [
        { icon: <Heart />, title: "Empathy First", desc: "We build technology that understands the anxiety of learning a new language." },
        { icon: <Target />, title: "Precision", desc: "Our AI doesn't guess. It analyzes audio waves at the millisecond level." },
        { icon: <Globe />, title: "Accessibility", desc: "Professional speech therapy should be available to everyone, everywhere." },
    ];

    return (
        <div className="hero-section-enterprise" style={{ minHeight: '100vh', display: 'block', paddingBottom: '4rem' }}>
            
            {/* 1. Header Section */}
            <div className="hero-container-enterprise" style={{ paddingTop: '6rem', textAlign: 'center' }}>
                <motion.div 
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                >
                    <span className="hero-badge-enterprise" style={{ margin: '0 auto 1rem' }}>
                        <Sparkles className="hero-badge-icon-enterprise" size={14} />
                        <span>Our Mission</span>
                    </span>
                    <h1 className="hero-headline-enterprise" style={{ fontSize: '3.5rem' }}>
                        Empowering the World to <br />
                        <span className="hero-headline-accent-enterprise">Speak with Confidence</span>
                    </h1>
                    <p className="hero-subheadline-enterprise" style={{ margin: '1.5rem auto', maxWidth: '700px' }}>
                        Pronunex bridges the gap between expensive speech therapy and independent learning. 
                        We believe that clear communication is a fundamental human right, not a luxury.
                    </p>
                </motion.div>
            </div>

            {/* 2. Stats Row */}
            <div className="hero-container-enterprise" style={{ marginTop: '4rem' }}>
                <div style={{ 
                    display: 'grid', 
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                    gap: '2rem',
                    background: 'white',
                    padding: '2rem',
                    borderRadius: '1rem',
                    boxShadow: '0 4px 20px rgba(0,0,0,0.05)'
                }}>
                    {stats.map((stat, idx) => (
                        <div key={idx} style={{ textAlign: 'center' }}>
                            <h3 style={{ fontSize: '2.5rem', fontWeight: '800', color: '#10b981', marginBottom: '0.5rem' }}>{stat.value}</h3>
                            <p style={{ color: '#64748b', fontWeight: '500' }}>{stat.label}</p>
                        </div>
                    ))}
                </div>
            </div>

            {/* 3. Values Grid */}
            <div className="hero-container-enterprise" style={{ marginTop: '6rem' }}>
                <h2 style={{ fontSize: '2rem', fontWeight: '700', color: '#0f172a', marginBottom: '3rem', textAlign: 'center' }}>
                    Core Values
                </h2>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                    {values.map((item, idx) => (
                        <motion.div 
                            key={idx}
                            whileHover={{ y: -5 }}
                            style={{ 
                                padding: '2rem', 
                                background: 'rgba(255,255,255,0.8)', 
                                borderRadius: '1rem', 
                                border: '1px solid rgba(0,0,0,0.05)',
                                backdropFilter: 'blur(10px)'
                            }}
                        >
                            <div style={{ 
                                width: '50px', height: '50px', 
                                background: '#ecfdf5', borderRadius: '12px', 
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                color: '#059669', marginBottom: '1.5rem'
                            }}>
                                {item.icon}
                            </div>
                            <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#1e293b', marginBottom: '1rem' }}>{item.title}</h3>
                            <p style={{ color: '#64748b', lineHeight: '1.6' }}>{item.desc}</p>
                        </motion.div>
                    ))}
                </div>
            </div>
        </div>
    );
}