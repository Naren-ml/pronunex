/**
 * Landing Page Components Barrel Export
 * Lazy-loaded components for optimized above-fold loading
 */

// Eager load - Above the fold (critical path)
export { default as HeroSection } from './HeroSection';

// Lazy load - Below the fold (deferred)
export { default as HowItWorksSection } from './HowItWorksSection';
export { default as FeatureGridSection } from './FeatureGridSection';
export { default as DataInActionSection } from './DataInActionSection';
export { default as TechnicalTrustSection } from './TechnicalTrustSection';
export { default as CTASection } from './CTASection';
export { default as Footer } from './Footer';

// Skeleton components
export { default as SectionSkeleton } from './SectionSkeleton';
