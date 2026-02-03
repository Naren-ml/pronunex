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

// Import your NEW pages
import AboutUs from './pages/AboutUs';
import Docs from './pages/Docs';
import Pricing from './pages/Pricing'; // Added this back
import Contact from './pages/Contact'; // Added this back
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
                {/* --- AUTH ROUTES --- */}
                <Route path="/login" element={<PublicRoute><Login /></PublicRoute>} />
                <Route path="/signup" element={<PublicRoute><Signup /></PublicRoute>} />
                <Route path="/forgot-password" element={<ForgotPassword />} />
                <Route path="/reset-password" element={<ResetPassword />} />

                {/* --- LANDING PAGE --- */}
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

                {/* --- PUBLIC INFO PAGES --- */}
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
                <Route 
                    path="/pricing" 
                    element={
                        <MainLayout>
                            <Pricing />
                        </MainLayout>
                    } 
                />
                <Route 
                    path="/contact" 
                    element={
                        <MainLayout>
                            <Contact />
                        </MainLayout>
                    } 
                />

                {/* --- PROTECTED ROUTES --- */}
                <Route path="/dashboard" element={<ProtectedRoute><MainLayout><Dashboard /></MainLayout></ProtectedRoute>} />
                <Route path="/phonemes" element={<ProtectedRoute><MainLayout><Phonemes /></MainLayout></ProtectedRoute>} />
                <Route path="/profile" element={<ProtectedRoute><MainLayout><Profile /></MainLayout></ProtectedRoute>} />
                
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

export default App;