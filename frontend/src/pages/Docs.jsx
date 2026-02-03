import React, { useState } from 'react';

const Docs = () => {
  const [activeSection, setActiveSection] = useState('getting-started');

  const sections = [
    { id: 'getting-started', title: 'Getting Started' },
    { id: 'installation', title: 'Installation' },
    { id: 'usage', title: 'Usage Guide' },
    { id: 'api', title: 'API Reference' },
    { id: 'faq', title: 'FAQ' },
  ];

  return (
    <div style={styles.container}>
      {/* Sidebar Navigation */}
      <div style={styles.sidebar}>
        <h3 style={styles.sidebarTitle}>Documentation</h3>
        <ul style={styles.navList}>
          {sections.map((section) => (
            <li 
              key={section.id} 
              onClick={() => setActiveSection(section.id)}
              style={{
                ...styles.navItem,
                ...(activeSection === section.id ? styles.activeNavItem : {})
              }}
            >
              {section.title}
            </li>
          ))}
        </ul>
      </div>

      {/* Main Content Area */}
      <div style={styles.content}>
        {activeSection === 'getting-started' && (
          <div>
            <h1 style={styles.title}>Getting Started</h1>
            <p style={styles.text}>
              Welcome to the Pronunex documentation. Pronunex is an advanced speech therapy tool 
              that uses AI to analyze your pronunciation in real-time.
            </p>
            <div style={styles.codeBlock}>
              npm install pronunex-sdk
            </div>
          </div>
        )}

        {activeSection === 'installation' && (
          <div>
            <h1 style={styles.title}>Installation</h1>
            <p style={styles.text}>
              To integrate Pronunex into your React application, follow these steps:
            </p>
            <ol style={styles.list}>
              <li>Ensure you have Node.js installed.</li>
              <li>Install the package via NPM or Yarn.</li>
              <li>Import the provider in your root component.</li>
            </ol>
          </div>
        )}

        {activeSection === 'usage' && (
          <div>
            <h1 style={styles.title}>Usage Guide</h1>
            <p style={styles.text}>
              Learn how to record audio and get feedback instantly.
            </p>
            <p style={styles.text}>
              1. Navigate to the <strong>Practice</strong> tab.<br/>
              2. Click the microphone icon.<br/>
              3. Read the sentence displayed on the screen.
            </p>
          </div>
        )}
        
        {/* Fallback for other sections */}
        {(activeSection === 'api' || activeSection === 'faq') && (
          <div>
            <h1 style={styles.title}>{sections.find(s => s.id === activeSection)?.title}</h1>
            <p style={styles.text}>Documentation for this section is coming soon.</p>
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    paddingTop: '80px', // Clear fixed navbar
    minHeight: '100vh',
    display: 'flex',
    background: '#f8fafc',
  },
  sidebar: {
    width: '250px',
    background: 'white',
    padding: '30px',
    borderRight: '1px solid #e2e8f0',
    position: 'fixed',
    bottom: 0,
    top: '80px', // Start below navbar
    overflowY: 'auto',
  },
  sidebarTitle: {
    fontSize: '1.2rem',
    marginBottom: '20px',
    color: '#334155',
  },
  navList: {
    listStyle: 'none',
    padding: 0,
  },
  navItem: {
    padding: '10px 15px',
    marginBottom: '5px',
    borderRadius: '8px',
    cursor: 'pointer',
    color: '#64748b',
    fontSize: '0.95rem',
    transition: 'all 0.2s',
  },
  activeNavItem: {
    background: '#e0e7ff',
    color: '#4f46e5',
    fontWeight: '600',
  },
  content: {
    marginLeft: '250px', // Offset for sidebar
    padding: '40px 60px',
    flex: 1,
    maxWidth: '800px',
  },
  title: {
    fontSize: '2.5rem',
    color: '#1e293b',
    marginBottom: '20px',
  },
  text: {
    fontSize: '1.1rem',
    color: '#475569',
    lineHeight: '1.7',
    marginBottom: '20px',
  },
  codeBlock: {
    background: '#1e293b',
    color: '#cbd5e1',
    padding: '15px',
    borderRadius: '8px',
    fontFamily: 'monospace',
    marginTop: '20px',
  },
  list: {
    marginLeft: '20px',
    color: '#475569',
    lineHeight: '1.8',
  }
};

export default Docs;