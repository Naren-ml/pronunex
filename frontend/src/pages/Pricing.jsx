import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { X, Check, AlertCircle } from 'lucide-react'; // Ensure you have lucide-react installed

const Pricing = () => {
  const [selectedPlan, setSelectedPlan] = useState(0); // Default to Base (index 0)
  const [showWaitlistModal, setShowWaitlistModal] = useState(false);
  const [waitlistPlan, setWaitlistPlan] = useState('');

  const handlePlanClick = (index, plan) => {
    setSelectedPlan(index);
  };

  const handleActionClick = (plan) => {
    if (plan.status === 'coming-soon') {
      setWaitlistPlan(plan.title);
      setShowWaitlistModal(true);
    }
  };

  const plans = [
    {
      id: 'base',
      title: 'Base',
      price: '₹0',
      period: 'Forever',
      description: 'Best for: New users exploring AI-based pronunciation improvement.',
      features: [
        'Limited pronunciation assessments',
        'Word-level accuracy scoring',
        'Detection of mispronounced words',
        'Standard practice exercises',
        'Basic progress tracking',
        'Web access'
      ],
      buttonText: 'Get Started',
      buttonLink: '/signup',
      status: 'active',
      ctaAction: 'link'
    },
    {
      id: 'pro',
      title: 'Pro',
      price: '₹399',
      period: '/ month',
      description: 'For: Serious learners seeking structured improvement.',
      features: [
        'Unlimited pronunciation assessments',
        'Phoneme-level error detection',
        'Detailed pronunciation & fluency feedback',
        'Adaptive practice recommendations',
        'Full progress dashboard',
        'Faster AI processing'
      ],
      buttonText: 'Notify Me',
      status: 'coming-soon',
      ctaAction: 'modal'
    },
    {
      id: 'pro-plus',
      title: 'Pro Plus',
      price: '₹799',
      period: '/ month',
      description: 'For: Professionals and advanced users.',
      features: [
        'Advanced pronunciation scoring models',
        'Personalized AI learning paths',
        'Long-term analytics & performance insights',
        'Exportable progress reports (PDF/CSV)',
        'Priority feature access & support'
      ],
      buttonText: 'Join Waitlist',
      status: 'coming-soon',
      ctaAction: 'modal'
    }
  ];

  return (
    <div style={styles.container}>
      {/* Header Section */}
      <div style={styles.header}>
        <h1 style={styles.title}>Simple, Transparent Pricing</h1>
        <p style={styles.subtitle}>
          Start your journey for free. Upgrade when you're ready for professional tools.
        </p>
        
        {/* Development Notice */}
        <div style={styles.devNotice}>
          <AlertCircle size={18} style={{ marginRight: '8px' }} />
          <span>
            <strong>Note:</strong> Pro plans are currently under development. 
            Join the waitlist to get early access and launch discounts.
          </span>
        </div>
      </div>

      {/* Pricing Grid */}
      <div style={styles.grid}>
        {plans.map((plan, index) => {
          const isSelected = selectedPlan === index;
          const isComingSoon = plan.status === 'coming-soon';

          return (
            <div 
              key={index} 
              onClick={() => handlePlanClick(index, plan)}
              style={{
                ...styles.card,
                ...(isSelected && !isComingSoon ? styles.selectedCard : {}),
                ...(isComingSoon ? styles.comingSoonCard : {})
              }}
            >
              {/* Badges */}
              {isSelected && !isComingSoon && <span style={styles.badge}>Selected Plan</span>}
              {isComingSoon && <span style={styles.devBadge}>🚧 In Development</span>}
              
              <h3 style={styles.planTitle}>{plan.title}</h3>
              
              <div style={styles.priceContainer}>
                <span style={styles.price}>{plan.price}</span>
                {plan.period && <span style={styles.period}>{plan.period}</span>}
              </div>
              
              <p style={styles.description}>{plan.description}</p>
              
              <ul style={styles.featuresList}>
                {plan.features.map((feature, idx) => (
                  <li key={idx} style={styles.featureItem}>
                    <Check size={16} style={{ 
                      color: isComingSoon ? '#94a3b8' : (isSelected ? '#4f46e5' : '#444'), 
                      marginRight: '8px',
                      flexShrink: 0 
                    }} />
                    {feature}
                  </li>
                ))}
              </ul>

              {/* Action Button */}
              {plan.ctaAction === 'link' ? (
                <Link 
                  to={plan.buttonLink} 
                  style={{
                    ...styles.button,
                    ...(isSelected ? styles.selectedButton : styles.outlineButton)
                  }}
                >
                  {plan.buttonText}
                </Link>
              ) : (
                <button
                  onClick={(e) => {
                    e.stopPropagation(); // Prevent card selection logic if needed
                    handleActionClick(plan);
                  }}
                  style={{
                    ...styles.button,
                    ...styles.waitlistButton
                  }}
                >
                  {plan.buttonText}
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Simple Waitlist Modal */}
      {showWaitlistModal && (
        <div style={styles.modalOverlay}>
          <div style={styles.modalContent}>
            <button 
              onClick={() => setShowWaitlistModal(false)}
              style={styles.closeModalBtn}
            >
              <X size={20} />
            </button>
            
            <h3 style={styles.modalTitle}>Join the {waitlistPlan} Waitlist</h3>
            <p style={styles.modalText}>
              Be the first to know when we launch! Early birds get <strong>20% off</strong>.
            </p>
            
            <form onSubmit={(e) => {
              e.preventDefault();
              alert(`Thanks! You've been added to the ${waitlistPlan} waitlist.`);
              setShowWaitlistModal(false);
            }} style={styles.modalForm}>
              <input 
                type="email" 
                placeholder="Enter your email" 
                required 
                style={styles.modalInput}
              />
              <button type="submit" style={styles.modalSubmitBtn}>
                Notify Me
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};

// Styles
const styles = {
  container: {
    padding: '120px 20px 60px',
    minHeight: '100vh',
    background: 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
    display: 'flex',
    flexDirection: 'column',
    alignItems: 'center'
  },
  header: {
    marginBottom: '50px',
    textAlign: 'center',
    maxWidth: '800px'
  },
  title: {
    fontSize: '2.5rem',
    color: '#1e293b',
    marginBottom: '10px',
    fontFamily: '"Inter", sans-serif',
    fontWeight: '800'
  },
  subtitle: {
    fontSize: '1.2rem',
    color: '#475569',
    marginBottom: '20px'
  },
  devNotice: {
    background: '#e0e7ff',
    color: '#4338ca',
    padding: '12px 20px',
    borderRadius: '50px',
    fontSize: '0.9rem',
    display: 'inline-flex',
    alignItems: 'center',
    border: '1px solid #c7d2fe'
  },
  grid: {
    display: 'flex',
    justifyContent: 'center',
    gap: '30px',
    flexWrap: 'wrap',
    maxWidth: '1200px',
    width: '100%'
  },
  card: {
    background: 'rgba(255, 255, 255, 0.8)',
    backdropFilter: 'blur(12px)',
    borderRadius: '24px',
    padding: '40px',
    width: '320px',
    boxShadow: '0 4px 6px rgba(0, 0, 0, 0.05)',
    border: '2px solid transparent',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)',
    cursor: 'pointer',
    textAlign: 'left'
  },
  // "Coming Soon" specific style
  comingSoonCard: {
    background: 'rgba(241, 245, 249, 0.6)', // Greyed out background
    borderColor: '#e2e8f0',
    opacity: 0.9,
    cursor: 'default' // Change cursor to default since selection is less relevant
  },
  selectedCard: {
    borderColor: '#4f46e5',
    background: 'rgba(255, 255, 255, 0.95)',
    transform: 'translateY(-10px) scale(1.02)',
    boxShadow: '0 20px 25px -5px rgba(79, 70, 229, 0.1), 0 10px 10px -5px rgba(79, 70, 229, 0.04)',
    zIndex: 2
  },
  badge: {
    position: 'absolute',
    top: '-12px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: '#4f46e5',
    color: 'white',
    padding: '6px 16px',
    borderRadius: '20px',
    fontSize: '0.75rem',
    fontWeight: '700',
    textTransform: 'uppercase',
    boxShadow: '0 4px 6px rgba(79, 70, 229, 0.2)'
  },
  devBadge: {
    position: 'absolute',
    top: '-12px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: '#f59e0b', // Amber/Orange for warning/construction
    color: 'white',
    padding: '6px 16px',
    borderRadius: '20px',
    fontSize: '0.75rem',
    fontWeight: '700',
    textTransform: 'uppercase',
    boxShadow: '0 4px 6px rgba(245, 158, 11, 0.2)'
  },
  planTitle: {
    fontSize: '1.5rem',
    marginBottom: '8px',
    color: '#1e293b',
    fontWeight: '700'
  },
  priceContainer: {
    display: 'flex',
    alignItems: 'baseline',
    marginBottom: '16px'
  },
  price: {
    fontSize: '2.5rem',
    fontWeight: '800',
    color: '#0f172a',
    letterSpacing: '-0.02em'
  },
  period: {
    fontSize: '1rem',
    color: '#64748b',
    marginLeft: '5px'
  },
  description: {
    color: '#64748b',
    marginBottom: '32px',
    fontSize: '0.95rem',
    lineHeight: '1.5',
    minHeight: '42px' // Keeps cards aligned even with different text lengths
  },
  featuresList: {
    listStyle: 'none',
    padding: 0,
    margin: '0 0 32px',
    flexGrow: 1 // Pushes button to bottom
  },
  featureItem: {
    marginBottom: '12px',
    color: '#334155',
    fontSize: '0.95rem',
    display: 'flex',
    alignItems: 'start',
    lineHeight: '1.4'
  },
  button: {
    marginTop: 'auto',
    display: 'block',
    padding: '12px 24px',
    borderRadius: '12px',
    textAlign: 'center',
    textDecoration: 'none',
    fontWeight: '600',
    transition: 'all 0.2s',
    border: 'none',
    cursor: 'pointer',
    fontSize: '1rem'
  },
  selectedButton: {
    background: '#4f46e5',
    color: 'white',
    boxShadow: '0 4px 6px rgba(79, 70, 229, 0.25)'
  },
  outlineButton: {
    background: 'transparent',
    color: '#4f46e5',
    border: '2px solid #e2e8f0'
  },
  waitlistButton: {
    background: '#e2e8f0',
    color: '#64748b',
    border: '1px solid #cbd5e1'
  },
  // Modal Styles
  modalOverlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0,0,0,0.5)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
    backdropFilter: 'blur(5px)'
  },
  modalContent: {
    background: 'white',
    padding: '40px',
    borderRadius: '24px',
    width: '90%',
    maxWidth: '400px',
    position: 'relative',
    textAlign: 'center',
    boxShadow: '0 20px 25px -5px rgba(0, 0, 0, 0.1)'
  },
  closeModalBtn: {
    position: 'absolute',
    top: '15px',
    right: '15px',
    background: 'transparent',
    border: 'none',
    cursor: 'pointer',
    color: '#64748b'
  },
  modalTitle: {
    fontSize: '1.5rem',
    color: '#1e293b',
    marginBottom: '10px'
  },
  modalText: {
    color: '#64748b',
    marginBottom: '20px'
  },
  modalForm: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px'
  },
  modalInput: {
    padding: '12px',
    borderRadius: '8px',
    border: '1px solid #e2e8f0',
    fontSize: '1rem'
  },
  modalSubmitBtn: {
    padding: '12px',
    borderRadius: '8px',
    border: 'none',
    background: '#4f46e5',
    color: 'white',
    fontWeight: '600',
    cursor: 'pointer',
    fontSize: '1rem'
  }
};

export default Pricing;