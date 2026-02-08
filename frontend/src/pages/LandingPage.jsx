/**
 * Pronunex Landing Page - Enterprise Grade
 * Optimized with lazy loading for fast initial load
 * Hero section loads immediately, other sections load progressively
 */

import { Suspense, lazy, useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

// Eager load - Above the fold (critical path)
import HeroSection from '../components/landing/HeroSection';
import SectionSkeleton from '../components/landing/SectionSkeleton';

// Lazy load - Below the fold (deferred)
const HowItWorksSection = lazy(() => import('../components/landing/HowItWorksSection'));
const FeatureGridSection = lazy(() => import('../components/landing/FeatureGridSection'));
const DataInActionSection = lazy(() => import('../components/landing/DataInActionSection'));
const TechnicalTrustSection = lazy(() => import('../components/landing/TechnicalTrustSection'));
const CTASection = lazy(() => import('../components/landing/CTASection'));
const Footer = lazy(() => import('../components/landing/Footer'));

import './LandingPage.css';

/**
 * Main Landing Page Component
 * Progressive loading strategy:
 * 1. Hero section renders immediately (viewport content)
 * 2. Below-fold sections lazy load with skeleton fallbacks
 */
export default function LandingPage() {
    const [isVisible, setIsVisible] = useState(false);
    const [shouldLoadBelow, setShouldLoadBelow] = useState(false);

    useEffect(() => {
        // Trigger initial visibility animation
        setIsVisible(true);

        // Delay loading below-fold content slightly for smooth hero experience
        const timer = setTimeout(() => {
            setShouldLoadBelow(true);
        }, 300);

        return () => clearTimeout(timer);
    }, []);

    // Handle hash scroll when navigating from other pages
    useEffect(() => {
        if (shouldLoadBelow && window.location.hash) {
            const targetId = window.location.hash.replace('#', '');
            // Small delay to ensure component is rendered
            const scrollTimer = setTimeout(() => {
                const element = document.getElementById(targetId);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                }
            }, 100);
            return () => clearTimeout(scrollTimer);
        }
    }, [shouldLoadBelow]);

    return (
        <AnimatePresence>
            <motion.div
                className={`landing-page-enterprise ${isVisible ? 'visible' : ''}`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3 }}
            >
                {/* Above the fold - Loads immediately */}
                <HeroSection />

                {/* Below the fold - Lazy loaded with skeletons */}
                {shouldLoadBelow && (
                    <>
                        <Suspense fallback={<SectionSkeleton type="steps" height="500px" />}>
                            <HowItWorksSection />
                        </Suspense>

                        <Suspense fallback={<SectionSkeleton type="cards" height="500px" />}>
                            <FeatureGridSection />
                        </Suspense>

                        <Suspense fallback={<SectionSkeleton type="data" height="500px" />}>
                            <DataInActionSection />
                        </Suspense>

                        <Suspense fallback={<SectionSkeleton type="default" height="400px" />}>
                            <TechnicalTrustSection />
                        </Suspense>

                        <Suspense fallback={<SectionSkeleton type="default" height="300px" />}>
                            <CTASection />
                        </Suspense>

                        <Suspense fallback={<SectionSkeleton type="default" height="300px" />}>
                            <Footer />
                        </Suspense>
                    </>
                )}
            </motion.div>
        </AnimatePresence>
    );
}
