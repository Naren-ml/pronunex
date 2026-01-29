/**
 * Settings Page
 * Comprehensive settings with Appearance, Notifications, Audio, Practice, and Privacy sections
 */

import React, { useState } from 'react';
import { useSettings } from '../context/SettingsContext';
import { useAuth } from '../context/AuthContext';
import { useUI } from '../context/UIContext';
import { DEFAULT_SETTINGS } from '../config/globalConfig';
import {
    Monitor, Moon, Sun, Type, Check, RefreshCw,
    Bell, Volume2, Mic, Target, Shield, Download,
    Trash2, UserX, Clock, Minus, Plus
} from 'lucide-react';
import './SettingsPage.css';

/**
 * Toggle Switch Component
 */
const ToggleSwitch = ({ checked, onChange, id }) => (
    <label className="toggle-switch">
        <input
            type="checkbox"
            id={id}
            className="toggle-switch__input"
            checked={checked}
            onChange={(e) => onChange(e.target.checked)}
        />
        <span className="toggle-switch__slider"></span>
    </label>
);

/**
 * Number Stepper Component for Daily Goal
 */
const NumberStepper = ({ value, onChange, min = 1, max = 50, step = 5, label }) => (
    <div className="settings__number-input">
        <button
            type="button"
            className="settings__number-btn"
            onClick={() => onChange(Math.max(min, value - step))}
            disabled={value <= min}
            aria-label="Decrease"
        >
            <Minus size={16} />
        </button>
        <span className="settings__number-value">{value}</span>
        <button
            type="button"
            className="settings__number-btn"
            onClick={() => onChange(Math.min(max, value + step))}
            disabled={value >= max}
            aria-label="Increase"
        >
            <Plus size={16} />
        </button>
        {label && <span className="settings__number-label">{label}</span>}
    </div>
);

const SettingsPage = () => {
    const { settings, updateSetting, resetSettings, presets } = useSettings();
    const { toast } = useUI();
    const { logout } = useAuth();

    const [isTesting, setIsTesting] = useState(false);

    /**
     * Test Microphone
     */
    const handleTestMic = async () => {
        try {
            setIsTesting(true);
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            toast.success('Microphone is working correctly');
            // Stop the stream after test
            setTimeout(() => {
                stream.getTracks().forEach(track => track.stop());
                setIsTesting(false);
            }, 2000);
        } catch (error) {
            toast.error('Microphone access denied or not available');
            setIsTesting(false);
        }
    };

    /**
     * Handle Export Data (placeholder)
     */
    const handleExportData = () => {
        toast.info('Export feature coming soon');
    };

    return (
        <div className="settings animate-fade-in">
            {/* Header */}
            <header className="settings__header">
                <h1 className="settings__title">Settings</h1>
                <p className="settings__subtitle">
                    Manage your application preferences and system configuration.
                </p>
            </header>

            {/* =========================================
                Appearance Section
                ========================================= */}
            <section className="settings__section">
                <div className="settings__section-header">
                    <div className="settings__section-icon settings__section-icon--appearance">
                        <Monitor size={20} />
                    </div>
                    <div className="settings__section-info">
                        <h2>Appearance</h2>
                        <p>Customize how Pronunex looks and feels.</p>
                    </div>
                </div>
                <div className="settings__section-body">
                    {/* Theme Toggle */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Theme Mode</h3>
                            <p>Select your preferred interface theme.</p>
                        </div>
                        <div className="settings__row-control">
                            <div className="settings__theme-buttons">
                                <button
                                    type="button"
                                    onClick={() => updateSetting('theme', 'light')}
                                    className={`settings__theme-btn ${settings.theme === 'light' ? 'settings__theme-btn--active' : ''}`}
                                >
                                    <Sun size={18} />
                                    <span>Light</span>
                                </button>
                                <button
                                    type="button"
                                    onClick={() => updateSetting('theme', 'dark')}
                                    className={`settings__theme-btn ${settings.theme === 'dark' ? 'settings__theme-btn--active' : ''}`}
                                >
                                    <Moon size={18} />
                                    <span>Dark</span>
                                </button>
                            </div>
                        </div>
                    </div>

                    {/* Accent Color */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Accent Color</h3>
                            <p>Choose a primary brand color.</p>
                        </div>
                        <div className="settings__row-control">
                            <div className="settings__color-presets">
                                {presets.map((color) => (
                                    <button
                                        key={color.name}
                                        type="button"
                                        onClick={() => updateSetting('primaryColor', color.value)}
                                        className={`settings__color-btn ${settings.primaryColor === color.value ? 'settings__color-btn--active' : ''}`}
                                        style={{ backgroundColor: color.value }}
                                        aria-label={`Select ${color.name} color`}
                                    >
                                        {settings.primaryColor === color.value && (
                                            <Check size={18} className="settings__color-check" />
                                        )}
                                        <span className="settings__color-name">{color.name}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Font Scale */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Font Scale</h3>
                            <p>Adjust the base text size.</p>
                        </div>
                        <div className="settings__row-control">
                            <Type size={14} style={{ color: 'var(--color-gray-400)' }} />
                            <div className="settings__slider-container">
                                <input
                                    type="range"
                                    min="75"
                                    max="125"
                                    step="5"
                                    value={settings.fontScale}
                                    onChange={(e) => updateSetting('fontScale', parseInt(e.target.value))}
                                    className="settings__slider"
                                />
                                <span className="settings__slider-value">{settings.fontScale}%</span>
                            </div>
                            <Type size={22} style={{ color: 'var(--color-gray-700)' }} />
                        </div>
                    </div>
                </div>
            </section>

            {/* =========================================
                Notifications Section
                ========================================= */}
            <section className="settings__section">
                <div className="settings__section-header">
                    <div className="settings__section-icon settings__section-icon--notifications">
                        <Bell size={20} />
                    </div>
                    <div className="settings__section-info">
                        <h2>Notifications</h2>
                        <p>Configure reminders and alerts.</p>
                    </div>
                </div>
                <div className="settings__section-body">
                    {/* Email Reminders */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Email Reminders</h3>
                            <p>Receive practice reminders via email.</p>
                        </div>
                        <div className="settings__row-control">
                            <ToggleSwitch
                                id="emailReminders"
                                checked={settings.emailReminders}
                                onChange={(val) => updateSetting('emailReminders', val)}
                            />
                        </div>
                    </div>

                    {/* Browser Notifications */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Browser Notifications</h3>
                            <p>Enable push notifications in your browser.</p>
                        </div>
                        <div className="settings__row-control">
                            <ToggleSwitch
                                id="browserNotifications"
                                checked={settings.browserNotifications}
                                onChange={(val) => updateSetting('browserNotifications', val)}
                            />
                        </div>
                    </div>

                    {/* Reminder Time */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Daily Reminder Time</h3>
                            <p>When should we remind you to practice?</p>
                        </div>
                        <div className="settings__row-control">
                            <Clock size={18} style={{ color: 'var(--color-gray-400)' }} />
                            <input
                                type="time"
                                value={settings.reminderTime}
                                onChange={(e) => updateSetting('reminderTime', e.target.value)}
                                className="settings__time-input"
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* =========================================
                Audio & Microphone Section
                ========================================= */}
            <section className="settings__section">
                <div className="settings__section-header">
                    <div className="settings__section-icon settings__section-icon--audio">
                        <Volume2 size={20} />
                    </div>
                    <div className="settings__section-info">
                        <h2>Audio & Microphone</h2>
                        <p>Adjust audio playback and microphone settings.</p>
                    </div>
                </div>
                <div className="settings__section-body">
                    {/* Microphone Sensitivity */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Microphone Sensitivity</h3>
                            <p>Adjust input volume threshold.</p>
                        </div>
                        <div className="settings__row-control">
                            <div className="settings__slider-container">
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="5"
                                    value={settings.micSensitivity}
                                    onChange={(e) => updateSetting('micSensitivity', parseInt(e.target.value))}
                                    className="settings__slider"
                                />
                                <span className="settings__slider-value">{settings.micSensitivity}%</span>
                            </div>
                        </div>
                    </div>

                    {/* Playback Volume */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Playback Volume</h3>
                            <p>Audio feedback and pronunciation playback.</p>
                        </div>
                        <div className="settings__row-control">
                            <div className="settings__slider-container">
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="5"
                                    value={settings.playbackVolume}
                                    onChange={(e) => updateSetting('playbackVolume', parseInt(e.target.value))}
                                    className="settings__slider"
                                />
                                <span className="settings__slider-value">{settings.playbackVolume}%</span>
                            </div>
                        </div>
                    </div>

                    {/* Test Microphone */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Test Microphone</h3>
                            <p>Check if your microphone is working.</p>
                        </div>
                        <div className="settings__row-control">
                            <button
                                type="button"
                                onClick={handleTestMic}
                                disabled={isTesting}
                                className={`settings__test-btn ${isTesting ? 'settings__test-btn--recording' : ''}`}
                            >
                                <Mic size={16} />
                                {isTesting ? 'Listening...' : 'Test Mic'}
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* =========================================
                Practice Preferences Section
                ========================================= */}
            <section className="settings__section">
                <div className="settings__section-header">
                    <div className="settings__section-icon settings__section-icon--practice">
                        <Target size={20} />
                    </div>
                    <div className="settings__section-info">
                        <h2>Practice Preferences</h2>
                        <p>Customize your learning experience.</p>
                    </div>
                </div>
                <div className="settings__section-body">
                    {/* Default Difficulty */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Default Difficulty</h3>
                            <p>Starting difficulty for new sessions.</p>
                        </div>
                        <div className="settings__row-control">
                            <select
                                value={settings.defaultDifficulty}
                                onChange={(e) => updateSetting('defaultDifficulty', e.target.value)}
                                className="settings__select"
                            >
                                <option value="beginner">Beginner</option>
                                <option value="intermediate">Intermediate</option>
                                <option value="advanced">Advanced</option>
                            </select>
                        </div>
                    </div>

                    {/* Session Length */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Session Length</h3>
                            <p>Preferred practice session duration.</p>
                        </div>
                        <div className="settings__row-control">
                            <select
                                value={settings.sessionLength}
                                onChange={(e) => updateSetting('sessionLength', parseInt(e.target.value))}
                                className="settings__select"
                            >
                                <option value={5}>5 minutes</option>
                                <option value={10}>10 minutes</option>
                                <option value={15}>15 minutes</option>
                                <option value={20}>20 minutes</option>
                                <option value={30}>30 minutes</option>
                            </select>
                        </div>
                    </div>

                    {/* Daily Goal */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Daily Goal</h3>
                            <p>Number of sentences to practice each day.</p>
                        </div>
                        <div className="settings__row-control">
                            <NumberStepper
                                value={settings.dailyGoal}
                                onChange={(val) => updateSetting('dailyGoal', val)}
                                min={5}
                                max={50}
                                step={5}
                                label="sentences"
                            />
                        </div>
                    </div>

                    {/* Auto-Advance */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Auto-Advance</h3>
                            <p>Automatically move to next sentence after recording.</p>
                        </div>
                        <div className="settings__row-control">
                            <ToggleSwitch
                                id="autoAdvance"
                                checked={settings.autoAdvance}
                                onChange={(val) => updateSetting('autoAdvance', val)}
                            />
                        </div>
                    </div>
                </div>
            </section>

            {/* =========================================
                Data & Privacy Section
                ========================================= */}
            <section className="settings__section">
                <div className="settings__section-header">
                    <div className="settings__section-icon settings__section-icon--privacy">
                        <Shield size={20} />
                    </div>
                    <div className="settings__section-info">
                        <h2>Data & Privacy</h2>
                        <p>Manage your data and account.</p>
                    </div>
                </div>
                <div className="settings__section-body">
                    {/* Export Data */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Export My Data</h3>
                            <p>Download all your practice data and progress.</p>
                        </div>
                        <div className="settings__row-control">
                            <button
                                type="button"
                                onClick={handleExportData}
                                className="settings__danger-btn"
                            >
                                <Download size={16} />
                                Export Data
                                <span className="settings__coming-soon">Coming Soon</span>
                            </button>
                        </div>
                    </div>

                    {/* Clear Practice History */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Clear Practice History</h3>
                            <p>Remove all your practice session records.</p>
                        </div>
                        <div className="settings__row-control">
                            <button
                                type="button"
                                disabled
                                className="settings__danger-btn"
                            >
                                <Trash2 size={16} />
                                Clear History
                                <span className="settings__coming-soon">Coming Soon</span>
                            </button>
                        </div>
                    </div>

                    {/* Delete Account */}
                    <div className="settings__row">
                        <div className="settings__row-label">
                            <h3>Delete Account</h3>
                            <p>Permanently delete your account and all data.</p>
                        </div>
                        <div className="settings__row-control">
                            <button
                                type="button"
                                disabled
                                className="settings__danger-btn settings__danger-btn--destructive"
                            >
                                <UserX size={16} />
                                Delete Account
                                <span className="settings__coming-soon">Coming Soon</span>
                            </button>
                        </div>
                    </div>
                </div>
            </section>

            {/* Footer / Reset */}
            <footer className="settings__footer">
                <button
                    type="button"
                    onClick={() => {
                        if (window.confirm('Are you sure you want to reset all settings to default?')) {
                            resetSettings();
                            toast.info('Settings restored to default');
                        }
                    }}
                    className="settings__reset-btn"
                >
                    <RefreshCw size={14} />
                    Reset to Defaults
                </button>
            </footer>
        </div>
    );
};

export default SettingsPage;
