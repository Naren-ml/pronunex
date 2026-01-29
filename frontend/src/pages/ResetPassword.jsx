/**
 * Reset Password Page
 * Enter new password with token from email
 */

import { useState, useEffect } from 'react';
import { Link, useNavigate, useSearchParams } from 'react-router-dom';
import { Lock, ArrowLeft, CheckCircle, Eye, EyeOff, Save } from 'lucide-react';
import { api } from '../api/client';
import { ENDPOINTS } from '../api/endpoints';
import { Spinner } from '../components/Loader';
import { useUI } from '../context/UIContext';
import './Auth.css';

export default function ResetPassword() {
    const navigate = useNavigate();
    const [searchParams] = useSearchParams();
    const token = searchParams.get('token');
    const { toast } = useUI();

    const [password, setPassword] = useState('');
    const [confirmPassword, setConfirmPassword] = useState('');
    const [showPassword, setShowPassword] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [isSuccess, setIsSuccess] = useState(false);
    const [errors, setErrors] = useState({});

    // Redirect if no token
    useEffect(() => {
        if (!token) {
            toast.error('Invalid password reset link.');
            navigate('/login');
        }
    }, [token, navigate, toast]);

    const validate = () => {
        const newErrors = {};
        if (!password) {
            newErrors.password = 'Password is required';
        } else if (password.length < 8) {
            newErrors.password = 'Password must be at least 8 characters';
        }

        if (password !== confirmPassword) {
            newErrors.confirmPassword = 'Passwords do not match';
        }

        setErrors(newErrors);
        return Object.keys(newErrors).length === 0;
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        if (!validate()) return;

        setIsLoading(true);
        setErrors({});

        try {
            await api.post(ENDPOINTS.AUTH.PASSWORD_RESET_CONFIRM, {
                token,
                password,
            });
            setIsSuccess(true);
            toast.success('Password reset successfully!');
        } catch (error) {
            const errorData = error.data || {};
            if (errorData.password) {
                setErrors({
                    password: Array.isArray(errorData.password)
                        ? errorData.password[0]
                        : errorData.password,
                });
            } else if (errorData.token) {
                setErrors({ form: 'Invalid or expired reset token.' });
            } else {
                setErrors({
                    form: error.message || 'Failed to reset password. Please try again.',
                });
            }
        } finally {
            setIsLoading(false);
        }
    };

    if (isSuccess) {
        return (
            <div className="auth-page">
                <div className="auth-container">
                    <div className="auth-header">
                        <Link to="/" className="auth-logo-link">
                            <div className="auth-logo">
                                <img src="/icon.png" alt="Pronunex" />
                            </div>
                        </Link>
                        <div className="auth-success-icon">
                            <CheckCircle size={48} />
                        </div>
                        <h1 className="auth-title">Password Reset!</h1>
                        <p className="auth-subtitle">
                            Your password has been securely updated.
                        </p>
                    </div>

                    <div className="auth-footer">
                        <Link to="/login" className="btn btn--primary w-full">
                            Sign In with New Password
                        </Link>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="auth-page">
            <div className="auth-container">
                <Link to="/login" className="auth-back-btn">
                    <ArrowLeft size={18} />
                    <span>Back to Login</span>
                </Link>

                <div className="auth-header">
                    <Link to="/" className="auth-logo-link">
                        <div className="auth-logo">
                            <img src="/icon.png" alt="Pronunex" />
                        </div>
                    </Link>
                    <h1 className="auth-title">Set New Password</h1>
                    <p className="auth-subtitle">
                        Create a strong password for your account
                    </p>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {errors.form && (
                        <div className="auth-error" role="alert">
                            {errors.form}
                        </div>
                    )}

                    <div className="form-group">
                        <label htmlFor="password" className="form-label">
                            New Password
                        </label>
                        <div className="input-wrapper">
                            <Lock className="input-icon" size={18} />
                            <input
                                type={showPassword ? 'text' : 'password'}
                                id="password"
                                className={`form-input ${errors.password ? 'form-input--error' : ''}`}
                                placeholder="Min. 8 characters"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                disabled={isLoading}
                            />
                            <button
                                type="button"
                                className="input-action"
                                onClick={() => setShowPassword(!showPassword)}
                            >
                                {showPassword ? <EyeOff size={18} /> : <Eye size={18} />}
                            </button>
                        </div>
                        {errors.password && (
                            <span className="form-error">{errors.password}</span>
                        )}
                    </div>

                    <div className="form-group">
                        <label htmlFor="confirmPassword" className="form-label">
                            Confirm Password
                        </label>
                        <div className="input-wrapper">
                            <Lock className="input-icon" size={18} />
                            <input
                                type={showPassword ? 'text' : 'password'}
                                id="confirmPassword"
                                className={`form-input ${errors.confirmPassword ? 'form-input--error' : ''}`}
                                placeholder="Re-enter password"
                                value={confirmPassword}
                                onChange={(e) => setConfirmPassword(e.target.value)}
                                disabled={isLoading}
                            />
                        </div>
                        {errors.confirmPassword && (
                            <span className="form-error">{errors.confirmPassword}</span>
                        )}
                    </div>

                    <button
                        type="submit"
                        className="auth-submit btn btn--primary btn--lg w-full"
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <>
                                <Spinner size="sm" />
                                <span>Updating...</span>
                            </>
                        ) : (
                            <>
                                <Save size={20} />
                                <span>Reset Password</span>
                            </>
                        )}
                    </button>
                </form>
            </div>
        </div>
    );
}
