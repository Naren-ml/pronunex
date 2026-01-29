/**
 * Main App Component
 * Router setup with lazy loading for heavy pages
 */

import { Suspense, lazy } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { useAuth } from './context/AuthContext';
import { Navbar } from './components/Navbar';
import { ToastContainer } from './components/Toast';
import { LoadingOverlay } from './components/Loader';
import { ErrorBoundary } from './components/ErrorBoundary';

// Eager loaded pages (small, frequently used)
import Login from './pages/Login';
import Signup from './pages/Signup';
import ForgotPassword from './pages/ForgotPassword';
import ResetPassword from './pages/ResetPassword';
import Dashboard from './pages/Dashboard';
import Phonemes from './pages/Phonemes';
import Profile from './pages/Profile';
import SettingsPage from './pages/SettingsPage';

// Lazy loaded pages (heavy components)
const Practice = lazy(() => import('./pages/Practice'));
const Progress = lazy(() => import('./pages/Progress'));
const AdminProfile = lazy(() => import('./pages/AdminProfile'));
const LandingPage = lazy(() => import('./pages/LandingPage'));

/**
 * Protected Route wrapper
 */
function ProtectedRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return <LoadingOverlay message="Loading..." />;
    }

    if (!isAuthenticated) {
        return <Navigate to="/login" replace />;
    }

    return children;
}

/**
 * Public Route wrapper (redirect to dashboard if already logged in)
 */
function PublicRoute({ children }) {
    const { isAuthenticated, isLoading } = useAuth();

    if (isLoading) {
        return <LoadingOverlay message="Loading..." />;
    }

    if (isAuthenticated) {
        return <Navigate to="/dashboard" replace />;
    }

    return children;
}

/**
 * Main layout with navbar
 */
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
                {/* Public routes */}
                <Route
                    path="/login"
                    element={
                        <PublicRoute>
                            <Login />
                        </PublicRoute>
                    }
                />
                <Route
                    path="/signup"
                    element={
                        <PublicRoute>
                            <Signup />
                        </PublicRoute>
                    }
                />
                <Route
                    path="/forgot-password"
                    element={<ForgotPassword />}
                />
                <Route
                    path="/reset-password"
                    element={<ResetPassword />}
                />

                {/* Landing page accessible to all users */}
                <Route
                    path="/"
                    element={
                        <Suspense fallback={<LoadingOverlay message="Loading..." />}>
                            <LandingPage />
                        </Suspense>
                    }
                />

                {/* Protected routes */}
                <Route
                    path="/dashboard"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Dashboard />
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/practice"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Suspense fallback={<LoadingOverlay message="Loading practice..." />}>
                                    <Practice />
                                </Suspense>
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/progress"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Suspense fallback={<LoadingOverlay message="Loading analytics..." />}>
                                    <Progress />
                                </Suspense>
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/phonemes"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Phonemes />
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/profile"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Profile />
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/settings"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <SettingsPage />
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />
                <Route
                    path="/admin"
                    element={
                        <ProtectedRoute>
                            <MainLayout>
                                <Suspense fallback={<LoadingOverlay message="Loading admin..." />}>
                                    <AdminProfile />
                                </Suspense>
                            </MainLayout>
                        </ProtectedRoute>
                    }
                />

                {/* Catch-all redirect */}
                <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
        </ErrorBoundary>
    );
}

export default App;
