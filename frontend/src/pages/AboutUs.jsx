import React from 'react';

const AboutUs = () => {
  return (
    <div className="about-container" style={styles.container}>
      <div style={styles.content}>
        <h1 style={styles.title}>About Pronunex</h1>
        <p style={styles.text}>
          Pronunex is an AI-powered speech therapy platform designed to help users improve their pronunciation
          and confidence. Our mission is to make speech therapy accessible, affordable, and effective for everyone.
        </p>
        
        <div style={styles.grid}>
          <div style={styles.card}>
            <h3>Our Mission</h3>
            <p>To democratize access to high-quality speech therapy tools through advanced AI technology.</p>
          </div>
          <div style={styles.card}>
            <h3>Our Vision</h3>
            <p>A world where communication barriers are bridged, and everyone can speak with confidence.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

const styles = {
  container: {
    padding: '120px 20px',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
    display: 'flex',
    justifyContent: 'center',
  },
  content: {
    maxWidth: '800px',
    textAlign: 'center',
  },
  title: {
    fontSize: '2.5rem',
    color: '#333',
    marginBottom: '20px',
  },
  text: {
    fontSize: '1.1rem',
    color: '#666',
    lineHeight: '1.6',
    marginBottom: '40px',
  },
  grid: {
    display: 'flex',
    gap: '20px',
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  card: {
    background: 'white',
    padding: '30px',
    borderRadius: '15px',
    flex: '1 1 300px',
    boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
  },
};

export default AboutUs;