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
import Dashboard from './pages/Dashboard';
import Phonemes from './pages/Phonemes';
import Profile from './pages/Profile';

// Import your NEW pages correctly
import AboutUs from './pages/AboutUs';
import Docs from './pages/Docs';
import Navbar from './components/landing/Navbar';

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

function MainLayout({ children }) {
    return (
        <div className="app-layout">
            <Navbar />
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
                {/* Auth Routes */}
                <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
                <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />

                {/* Landing Page */}
                <Route
                    path="/"
                    element={
                        <PublicRoute>
                            <Suspense fallback={<LoadingOverlay message="Loading..." />}>
                                <MainLayout>
                                    <LandingPage />
                                </MainLayout>
                            </Suspense>
                        </PublicRoute>
                    }
                />

                {/* --- CORRECTED INFO ROUTES --- */}
                {/* These now point to your real components, not the placeholder divs */}
                <Route 
                    path="/about" 
                    element={
                        <MainLayout>
                            <AboutUs />
                        </MainLayout>
                    } 
                />
                <Route 
                    path="/docs" 
                    element={
                        <MainLayout>
                            <Docs />
                        </MainLayout>
                    } 
                />

                {/* Protected routes */}
                <Route path="/dashboard" element={<ProtectedRoute><MainLayout><Dashboard /></MainLayout></ProtectedRoute>} />
                <Route path="/practice" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><Practice /></Suspense></MainLayout></ProtectedRoute>} />
                <Route path="/progress" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><Progress /></Suspense></MainLayout></ProtectedRoute>} />
                <Route path="/phonemes" element={<ProtectedRoute><MainLayout><Phonemes /></MainLayout></ProtectedRoute>} />
                <Route path="/profile" element={<ProtectedRoute><MainLayout><Profile /></MainLayout></ProtectedRoute>} />
                <Route path="/admin" element={<ProtectedRoute><MainLayout><Suspense fallback={<LoadingOverlay />}><AdminProfile /></Suspense></MainLayout></ProtectedRoute>} />

                {/* Catch-all redirect */}
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </ErrorBoundary>
    );
}

export default App;