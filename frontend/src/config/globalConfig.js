/**
 * Global Application Configuration & Defaults
 * Centralizes all user-configurable settings.
 */

export const DEFAULT_SETTINGS = {
    // Appearance
    theme: 'light', // 'light' | 'dark'
    primaryColor: '#059669', // Emerald-500
    fontScale: 100,

    // Notifications
    emailReminders: false,
    browserNotifications: false,
    reminderTime: '09:00',

    // Audio & Microphone
    micSensitivity: 50,
    playbackVolume: 80,

    // Practice Preferences
    defaultDifficulty: 'intermediate', // 'beginner' | 'intermediate' | 'advanced'
    sessionLength: 10, // minutes
    dailyGoal: 10, // number of sentences per day
    autoAdvance: true,

    // System
    apiBaseUrl: 'http://localhost:8000',
};

export const COLOR_PRESETS = [
    {
        name: 'Emerald', value: '#059669', shades: {
            50: '#ecfdf5', 100: '#d1fae5', 200: '#a7f3d0', 300: '#6ee7b7',
            400: '#34d399', 500: '#059669', 600: '#047857', 700: '#047857',
            800: '#064e3b', 900: '#064e3b'
        }
    },
    {
        name: 'Rose', value: '#e11d48', shades: {
            50: '#fff1f2', 100: '#ffe4e6', 200: '#fecdd3', 300: '#fda4af',
            400: '#fb7185', 500: '#f43f5e', 600: '#e11d48', 700: '#be123c',
            800: '#9f1239', 900: '#881337'
        }
    },
    {
        name: 'Amber', value: '#d97706', shades: {
            50: '#fffbeb', 100: '#fef3c7', 200: '#fde68a', 300: '#fcd34d',
            400: '#fbbf24', 500: '#f59e0b', 600: '#d97706', 700: '#b45309',
            800: '#92400e', 900: '#78350f'
        }
    },
    {
        name: 'Teal', value: '#0d9488', shades: {
            50: '#f0fdfa', 100: '#ccfbf1', 200: '#99f6e4', 300: '#5eead4',
            400: '#2dd4bf', 500: '#14b8a6', 600: '#0d9488', 700: '#0f766e',
            800: '#115e59', 900: '#134e4a'
        }
    }
];

// Keys for localStorage
export const STORAGE_KEYS = {
    SETTINGS: 'pronunex_settings_v1'
};
