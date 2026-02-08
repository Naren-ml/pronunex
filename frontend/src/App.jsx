/**
 * Main App Component
 * Router setup with lazy loading for heavy pages
 */


import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import { ToastContainer } from './components/Toast';
import { LoadingOverlay } from './components/Loader';
import { ErrorBoundary } from './components/ErrorBoundary';

// Eager loaded pages
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import Dashboard from './pages/Dashboard';
import Phonemes from './pages/Phonemes';
import Profile from './pages/Profile';
import SettingsPage from './pages/SettingsPage';

// Import your NEW pages
import AboutUs from './pages/AboutUs';
import Docs from './pages/Docs';
import Pricing from './pages/Pricing'; // Added this back
import Contact from './pages/Contact'; // Added this back
import LandingNavbar from './components/landing/Navbar';
import AuthNavbar from './components/Navbar';
import Features from './pages/Features';
import HowItWorks from './pages/HowItWorks';

// Lazy loaded pages
const Practice = lazy(() => import('./pages/Practice'));
const Progress = lazy(() => import('./pages/Progress'));
const AdminProfile = lazy(() => import('./pages/AdminProfile'));
const LandingPage = lazy(() => import('./pages/LandingPage'));

function ProtectedRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();
    if (isLoading) return <LoadingOverlay message="Loading..." />;
    if (!isAuthenticated) return <Navigate to="/login" replace />;
    return children;
}

function PublicRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();
    if (isLoading) return <LoadingOverlay message="Loading..." />;
    if (isAuthenticated) return <Navigate to="/dashboard" replace />;
    return children;
}

// Smart MainLayout - uses correct navbar based on auth state (for protected routes)
function MainLayout({ children }) {
    const { isAuthenticated } = useAuth();

    return (
        <div className="app-layout">
            {isAuthenticated ? <AuthNavbar /> : <LandingNavbar />}
            <main className="app-main">
                <ErrorBoundary>{children}</ErrorBoundary>
            </main>
        </div>
    );
}

// PublicLayout - always uses LandingNavbar (for public pages like landing, about, pricing)
function PublicLayout({ children }) {
    return (
        <div className="app-layout">
            <LandingNavbar />
            <main className="app-main">
                <ErrorBoundary>{children}</ErrorBoundary>
            </main>
        </div>
    );
}

function App() {
    return (
        <ErrorBoundary>
            <ToastContainer />
            <Routes>
                {/* --- AUTH ROUTES --- */}
                <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
                <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
                <Route path="/forgot-password" element={<ForgotPassword />} />
                <Route path="/reset-password" element={<ResetPassword />} />

                {/* --- LANDING PAGE --- */}
                <Route
                    path="/"
                    element={
                        <Suspense fallback={<LoadingOverlay message="Loading..." />}>
                            <PublicLayout>
                                <LandingPage />
                            </PublicLayout>
                        </Suspense>
                    }
                />

                {/* --- PUBLIC INFO PAGES --- */}
                <Route
                    path="/about"
                    element={
                        <PublicLayout>
                            <Contact />
                        </PublicLayout>
                    }
                />
                <Route
                    path="/features"
                    element={
                        <PublicLayout>
                            <Features />
                        </PublicLayout>
                    }
                />
                <Route
                    path="/how-it-works"
                    element={
                        <PublicLayout>
                            <HowItWorks />
                        </PublicLayout>
                    }
                />
                <Route
                    path="/pricing"
                    element={
                        <PublicLayout>
                            <Pricing />
                        </PublicLayout>
                    }
                />
                <Route
                    path="/contact"
                    element={
                        <PublicLayout>
                            <Contact />
                        </PublicLayout>
                    }
                />

                {/* --- PROTECTED ROUTES --- */}
                <Route path="/dashboard" element={<ProtectedRoute><MainLayout><Dashboard /></MainLayout></ProtectedRoute>} />
                <Route path="/phonemes" element={<ProtectedRoute><MainLayout><Phonemes /></MainLayout></ProtectedRoute>} />
                <Route path="/profile" element={<ProtectedRoute><MainLayout><Profile /></MainLayout></ProtectedRoute>} />
                <Route path="/settings" element={<ProtectedRoute><MainLayout><SettingsPage /></MainLayout></ProtectedRoute>} />

                {/* Lazy Loaded Protected Routes */}
                <Route path="/practice" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><Practice /></Suspense></MainLayout></ProtectedRoute>} />
                <Route path="/progress" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><Progress /></Suspense></MainLayout></ProtectedRoute>} />
                <Route path="/admin" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><AdminProfile /></Suspense></MainLayout></ProtectedRoute>} />

                {/* Catch-all redirect */}
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </ErrorBoundary>
    );
}

// Narendra Changes
import Games from "./pages/Games";

export default App;
