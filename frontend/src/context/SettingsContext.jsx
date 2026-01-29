import { createContext, useContext, useState, useEffect } from 'react';
import { DEFAULT_SETTINGS, STORAGE_KEYS, COLOR_PRESETS } from '../config/globalConfig';

const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
    // Load initial settings from storage or defaults
    const [settings, setSettings] = useState(() => {
        const stored = localStorage.getItem(STORAGE_KEYS.SETTINGS);
        if (stored) {
            try {
                return { ...DEFAULT_SETTINGS, ...JSON.parse(stored) };
            } catch (e) {
                console.error("Failed to parse settings", e);
                return DEFAULT_SETTINGS;
            }
        }
        return DEFAULT_SETTINGS;
    });

    // Save to storage whenever settings change
    useEffect(() => {
        localStorage.setItem(STORAGE_KEYS.SETTINGS, JSON.stringify(settings));
        applySettingsToDOM(settings);
    }, [settings]);

    // Apply settings to CSS Variables on the document root
    const applySettingsToDOM = (currentSettings) => {
        const root = document.documentElement;

        // 1. Theme (Light/Dark)
        document.documentElement.setAttribute('data-theme', currentSettings.theme);

        // 2. Font Scale
        // We adjust the base font-size of the html element. 
        // Default Tailwind base is usually 16px (100%).
        // 100% = 16px, 125% = 20px, etc.
        const basePixel = 16 * (currentSettings.fontScale / 100);
        root.style.fontSize = `${basePixel}px`;

        // 3. Primary Color Palette
        // Find the preset that matches the current primaryColor value, or default to Emerald
        const preset = COLOR_PRESETS.find(p => p.value === currentSettings.primaryColor) || COLOR_PRESETS[0];

        if (preset && preset.shades) {
            Object.entries(preset.shades).forEach(([shoe, value]) => {
                root.style.setProperty(`--color-primary-${shoe}`, value);
            });
        }
    };

    /**
     * Update a specific setting
     * @param {string} key - setting key (e.g., 'theme', 'primaryColor')
     * @param {any} value - new value
     */
    const updateSetting = (key, value) => {
        setSettings(prev => ({
            ...prev,
            [key]: value
        }));
    };

    /**
     * Reset all settings to defaults
     */
    const resetSettings = () => {
        setSettings(DEFAULT_SETTINGS);
    };

    const value = {
        settings,
        updateSetting,
        resetSettings,
        presets: COLOR_PRESETS
    };

    return (
        <SettingsContext.Provider value={value}>
            {children}
        </SettingsContext.Provider>
    );
}

export function useSettings() {
    const context = useContext(SettingsContext);
    if (!context) {
        throw new Error('useSettings must be used within a SettingsProvider');
    }
    return context;
}
