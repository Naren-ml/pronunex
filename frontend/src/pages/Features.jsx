import React from 'react';
import { Mic, Activity, Brain, Shield, Zap, Award } from 'lucide-react';

const Features = () => {
  const features = [
    {
      icon: <Mic size={32} color="#4f46e5" />,
      title: "Real-time AI Analysis",
      description: "Get instant feedback on your pronunciation as you speak. Our AI analyzes every phoneme in milliseconds."
    },
    {
      icon: <Activity size={32} color="#ec4899" />,
      title: "Detailed Scoring",
      description: "See exactly where you went wrong with word-level and phoneme-level accuracy scores."
    },
    {
      icon: <Brain size={32} color="#8b5cf6" />,
      title: "Adaptive Learning",
      description: "The platform learns your weak spots and suggests custom practice exercises to help you improve faster."
    },
    {
      icon: <Award size={32} color="#f59e0b" />,
      title: "Gamified Progress",
      description: "Earn badges and streaks as you practice. Visual charts help you track your improvement over time."
    },
    {
      icon: <Zap size={32} color="#10b981" />,
      title: "Instant Results",
      description: "No waiting. Our optimized backend processes audio instantly so you can practice without interruption."
    },
    {
      icon: <Shield size={32} color="#3b82f6" />,
      title: "Privacy First",
      description: "Your voice data is processed securely. We prioritize user privacy and data protection standards."
    }
  ];

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h1 style={styles.title}>Why Choose Pronunex?</h1>
        <p style={styles.subtitle}>
          Everything you need to master pronunciation, powered by advanced Artificial Intelligence.
        </p>
      </div>

      <div style={styles.grid}>
        {features.map((feature, index) => (
          <div key={index} style={styles.card}>
            <div style={styles.iconWrapper}>{feature.icon}</div>
            <h3 style={styles.cardTitle}>{feature.title}</h3>
            <p style={styles.cardText}>{feature.description}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '120px 20px 60px',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%)',
  },
  header: {
    textAlign: 'center',
    marginBottom: '60px',
    maxWidth: '800px',
    margin: '0 auto 60px',
  },
  title: {
    fontSize: '3rem',
    color: '#1e293b',
    marginBottom: '20px',
    fontWeight: '800',
  },
  subtitle: {
    fontSize: '1.2rem',
    color: '#64748b',
    lineHeight: '1.6',
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '30px',
    maxWidth: '1200px',
    margin: '0 auto',
  },
  card: {
    background: 'rgba(255, 255, 255, 0.7)',
    backdropFilter: 'blur(10px)',
    borderRadius: '20px',
    padding: '40px',
    border: '1px solid rgba(255, 255, 255, 0.8)',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
    transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    cursor: 'default',
  },
  iconWrapper: {
    background: 'white',
    width: '60px',
    height: '60px',
    borderRadius: '15px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: '20px',
    boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.05)',
  },
  cardTitle: {
    fontSize: '1.5rem',
    color: '#1e293b',
    marginBottom: '15px',
    fontWeight: '700',
  },
  cardText: {
    color: '#64748b',
    lineHeight: '1.6',
    fontSize: '1rem',
  }
};

export default Features;