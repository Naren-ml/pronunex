/**
 * ConfidenceMeter Component
 * Real-time confidence indicator during recording based on audio volume
 */

import { useState, useEffect, useRef } from 'react';
import { Volume2, VolumeX, Volume1 } from 'lucide-react';
import './ConfidenceMeter.css';

function ConfidenceMeter({ audioStream, isRecording }) {
    const [volume, setVolume] = useState(0);
    const [confidence, setConfidence] = useState('quiet');
    const animationRef = useRef(null);
    const analyserRef = useRef(null);
    const audioContextRef = useRef(null);

    useEffect(() => {
        if (!isRecording || !audioStream) {
            setVolume(0);
            setConfidence('quiet');
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
                audioContextRef.current = null;
            }
            return;
        }

        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        audioContextRef.current = audioContext;
        const analyser = audioContext.createAnalyser();
        analyser.fftSize = 256;
        analyserRef.current = analyser;

        const source = audioContext.createMediaStreamSource(audioStream);
        source.connect(analyser);

        const dataArray = new Uint8Array(analyser.frequencyBinCount);

        const updateVolume = () => {
            if (!isRecording) return;

            analyser.getByteFrequencyData(dataArray);

            // Calculate average volume
            const sum = dataArray.reduce((a, b) => a + b, 0);
            const avg = sum / dataArray.length;
            const normalizedVolume = Math.min(100, Math.round((avg / 128) * 100));

            setVolume(normalizedVolume);

            // Determine confidence level
            if (normalizedVolume < 15) {
                setConfidence('quiet');
            } else if (normalizedVolume < 40) {
                setConfidence('low');
            } else if (normalizedVolume < 70) {
                setConfidence('good');
            } else {
                setConfidence('loud');
            }

            animationRef.current = requestAnimationFrame(updateVolume);
        };

        updateVolume();

        return () => {
            if (animationRef.current) {
                cancelAnimationFrame(animationRef.current);
            }
            if (audioContextRef.current) {
                audioContextRef.current.close();
            }
        };
    }, [isRecording, audioStream]);

    const getVolumeIcon = () => {
        if (volume < 15) return VolumeX;
        if (volume < 50) return Volume1;
        return Volume2;
    };

    const VolumeIcon = getVolumeIcon();

    const getConfidenceLabel = () => {
        switch (confidence) {
            case 'quiet': return 'Speak louder';
            case 'low': return 'A bit louder';
            case 'good': return 'Perfect volume';
            case 'loud': return 'Too loud';
            default: return '';
        }
    };

    if (!isRecording) {
        return null;
    }

    return (
        <div className={`confidence-meter confidence-meter--${confidence}`}>
            <div className="confidence-meter__icon">
                <VolumeIcon size={20} />
            </div>
            <div className="confidence-meter__bar-container">
                <div
                    className="confidence-meter__bar"
                    style={{ width: `${volume}%` }}
                />
            </div>
            <span className="confidence-meter__label">{getConfidenceLabel()}</span>
        </div>
    );
}

export default ConfidenceMeter;
