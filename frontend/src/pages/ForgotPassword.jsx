/**
 * Forgot Password Page
 * Password reset request form
 */

import { useState } from 'react';
import { Link } from 'react-router-dom';
import { Mail, ArrowLeft, Send, CheckCircle } from 'lucide-react';
import { api } from '../api/client';
import { ENDPOINTS } from '../api/endpoints';
import { Spinner } from '../components/Loader';
import './Auth.css';

export function ForgotPassword() {
    const [email, setEmail] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [isSuccess, setIsSuccess] = useState(false);
    const [errors, setErrors] = useState({});

    const validate = () => {
        const newErrors = {};
        if (!email) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(email)) {
            newErrors.email = 'Invalid email format';
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
            await api.post(ENDPOINTS.AUTH.PASSWORD_RESET, { email });
            setIsSuccess(true);
        } catch (error) {
            // Parse backend validation errors
            const errorData = error.data || {};

            if (errorData.email) {
                setErrors({
                    email: Array.isArray(errorData.email) ? errorData.email[0] : errorData.email
                });
            } else if (errorData.error) {
                setErrors({ form: errorData.error });
            } else if (errorData.detail) {
                setErrors({ form: errorData.detail });
            } else {
                setErrors({ form: error.message || 'Failed to send reset email. Please try again.' });
            }
        } finally {
            setIsLoading(false);
        }
    };

    // Success state
    if (isSuccess) {
        return (
            <div className="auth-page">
                <div className="auth-container">
                    <Link to="/" className="auth-back-btn">
                        <ArrowLeft size={18} />
                        <span>Back</span>
                    </Link>

                    <div className="auth-header">
                        <Link to="/" className="auth-logo-link">
                            <div className="auth-logo">
                                <img src="/icon.png" alt="Pronunex" />
                            </div>
                        </Link>
                        <div className="auth-success-icon">
                            <CheckCircle size={48} />
                        </div>
                        <h1 className="auth-title">Check Your Email</h1>
                        <p className="auth-subtitle">
                            If an account exists with {email}, you will receive a password reset link shortly.
                        </p>
                    </div>

                    <div className="auth-footer">
                        <p>
                            Remember your password?{' '}
                            <Link to="/login" className="auth-link">
                                Sign in
                            </Link>
                        </p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="auth-page">
            <div className="auth-container">
                <Link to="/" className="auth-back-btn">
                    <ArrowLeft size={18} />
                    <span>Back</span>
                </Link>

                <div className="auth-header">
                    <Link to="/" className="auth-logo-link">
                        <div className="auth-logo">
                            <img src="/icon.png" alt="Pronunex" />
                        </div>
                    </Link>
                    <h1 className="auth-title">Forgot Password</h1>
                    <p className="auth-subtitle">
                        Enter your email and we will send you a reset link
                    </p>
                </div>

                <form className="auth-form" onSubmit={handleSubmit}>
                    {errors.form && (
                        <div className="auth-error" role="alert">
                            {errors.form}
                        </div>
                    )}

                    <div className="form-group">
                        <label htmlFor="email" className="form-label">
                            Email
                        </label>
                        <div className="input-wrapper">
                            <Mail className="input-icon" size={18} />
                            <input
                                type="email"
                                id="email"
                                name="email"
                                className={`form-input ${errors.email ? 'form-input--error' : ''}`}
                                placeholder="you@example.com"
                                value={email}
                                onChange={(e) => {
                                    setEmail(e.target.value);
                                    if (errors.email) {
                                        setErrors((prev) => ({ ...prev, email: null }));
                                    }
                                }}
                                autoComplete="email"
                                disabled={isLoading}
                            />
                        </div>
                        {errors.email && <span className="form-error">{errors.email}</span>}
                    </div>

                    <button
                        type="submit"
                        className="auth-submit btn btn--primary btn--lg w-full"
                        disabled={isLoading}
                    >
                        {isLoading ? (
                            <>
                                <Spinner size="sm" />
                                <span>Sending...</span>
                            </>
                        ) : (
                            <>
                                <Send size={20} />
                                <span>Send Reset Link</span>
                            </>
                        )}
                    </button>
                </form>

                <div className="auth-footer">
                    <p>
                        Remember your password?{' '}
                        <Link to="/login" className="auth-link">
                            Sign in
                        </Link>
                    </p>
                </div>
            </div>
        </div>
    );
}

export default ForgotPassword;
