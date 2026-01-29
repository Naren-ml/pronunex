import React from 'react';
import { BookOpen, Code, Terminal, FileText, ChevronRight, Search } from 'lucide-react';
import '../components/landing/HeroSection.css'; 

export default function Docs() {
    const guides = [
        { 
            title: 'Getting Started', 
            icon: <BookOpen size={20} />, 
            desc: 'Set up your account, microphone permissions, and run your first pronunciation assessment.',
            link: '#setup'
        },
        { 
            title: 'Phoneme API', 
            icon: <Code size={20} />, 
            desc: 'Understand how Pronunex scores individual phonemes and returns JSON analysis.',
            link: '#api'
        },
        { 
            title: 'Troubleshooting', 
            icon: <Terminal size={20} />, 
            desc: 'Common microphone issues, noise cancellation tips, and browser support.',
            link: '#help'
        },
        { 
            title: 'Data Privacy', 
            icon: <FileText size={20} />, 
            desc: 'How we process voice data locally and securely without storing raw audio.',
            link: '#privacy'
        }
    ];

    return (
        <div style={{ background: '#f8fafc', minHeight: '100vh', paddingTop: '6rem', paddingBottom: '4rem' }}>
            <div className="hero-container-enterprise">
                
                {/* Header with Search */}
                <div style={{ textAlign: 'center', marginBottom: '4rem' }}>
                    <h1 className="hero-headline-enterprise" style={{ fontSize: '2.5rem', marginBottom: '1rem' }}>
                        Documentation
                    </h1>
                    <p style={{ color: '#64748b', fontSize: '1.1rem', marginBottom: '2rem' }}>
                        Everything you need to master the Pronunex platform.
                    </p>
                    
                    {/* Fake Search Bar */}
                    <div style={{ 
                        maxWidth: '500px', margin: '0 auto', position: 'relative',
                        boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.1)'
                    }}>
                        <Search style={{ position: 'absolute', left: '1rem', top: '1rem', color: '#94a3b8' }} size={20} />
                        <input 
                            type="text" 
                            placeholder="Search guides, API, or phonemes..." 
                            style={{
                                width: '100%', padding: '1rem 1rem 1rem 3rem', borderRadius: '0.75rem',
                                border: '1px solid #e2e8f0', fontSize: '1rem', outline: 'none'
                            }}
                        />
                    </div>
                </div>

                {/* Guides Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '2rem' }}>
                    {guides.map((guide) => (
                        <div key={guide.title} className="doc-card" style={{
                            background: 'white', padding: '2rem', borderRadius: '1rem',
                            border: '1px solid #e2e8f0', transition: 'all 0.2s ease', cursor: 'pointer'
                        }}>
                            <div style={{ 
                                display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1rem',
                                color: '#10b981' 
                            }}>
                                {guide.icon}
                                <h3 style={{ fontSize: '1.25rem', fontWeight: '700', color: '#1e293b', margin: 0 }}>
                                    {guide.title}
                                </h3>
                            </div>
                            <p style={{ color: '#64748b', lineHeight: '1.6', marginBottom: '1.5rem' }}>
                                {guide.desc}
                            </p>
                            <div style={{ display: 'flex', alignItems: 'center', color: '#10b981', fontWeight: '600', fontSize: '0.9rem' }}>
                                Read Guide <ChevronRight size={16} />
                            </div>
                        </div>
                    ))}
                </div>

                {/* Code Snippet Example */}
                <div style={{ marginTop: '4rem', background: '#0f172a', borderRadius: '1rem', padding: '2rem', color: '#e2e8f0', overflow: 'hidden' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem', borderBottom: '1px solid #334155', paddingBottom: '1rem' }}>
                        <span style={{ fontFamily: 'monospace', color: '#34d399' }}>POST /api/analyze-audio</span>
                        <span style={{ fontSize: '0.8rem', opacity: 0.7 }}>JSON Response</span>
                    </div>
                    <pre style={{ margin: 0, fontFamily: 'monospace', fontSize: '0.9rem', lineHeight: '1.6' }}>
{`{
  "transcription": "The musician played beautiful melodies",
  "score": 85,
  "phonemes": [
    { "symbol": "th", "accuracy": 92, "status": "perfect" },
    { "symbol": "m",  "accuracy": 88, "status": "good" },
    { "symbol": "z",  "accuracy": 65, "status": "warning" }
  ]
}`}
</pre>
                </div>

            </div>
        </div>
    );
}