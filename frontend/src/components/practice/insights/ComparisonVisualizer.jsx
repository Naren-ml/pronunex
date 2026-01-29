import { useEffect, useRef, useState } from 'react';
import { Activity } from 'lucide-react';
import './ComparisonVisualizer.css'; // We'll assume simple styles or inline for canvas

export function ComparisonVisualizer({ userAudioUrl, referenceAudioUrl }) {
    const userCanvasRef = useRef(null);
    const refCanvasRef = useRef(null);

    // Helper to draw waveform
    const drawWaveform = async (url, canvas, color) => {
        if (!url || !canvas) return;

        try {
            const ctx = canvas.getContext('2d');
            const response = await fetch(url);
            const arrayBuffer = await response.arrayBuffer();
            const audioContext = new (window.AudioContext || window.webkitAudioContext)();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

            const rawData = audioBuffer.getChannelData(0);
            const samples = 200; // Number of bars
            const blockSize = Math.floor(rawData.length / samples);
            const filteredData = [];

            for (let i = 0; i < samples; i++) {
                let sum = 0;
                for (let j = 0; j < blockSize; j++) {
                    sum += Math.abs(rawData[blockSize * i + j]);
                }
                filteredData.push(sum / blockSize);
            }

            const width = canvas.width;
            const height = canvas.height;
            const barWidth = width / samples;
            const multiplier = height / Math.max(...filteredData);

            ctx.clearRect(0, 0, width, height);
            ctx.fillStyle = color;

            filteredData.forEach((val, i) => {
                const barHeight = val * multiplier * 0.9;
                ctx.fillRect(i * barWidth, (height - barHeight) / 2, barWidth - 1, barHeight);
            });

            audioContext.close();
        } catch (e) {
            console.error("Error visualizing audio:", e);
        }
    };

    useEffect(() => {
        if (userAudioUrl) drawWaveform(userAudioUrl, userCanvasRef.current, '#10b981');
        if (referenceAudioUrl) drawWaveform(referenceAudioUrl, refCanvasRef.current, '#2563eb');
    }, [userAudioUrl, referenceAudioUrl]);

    return (
        <div className="comparison-visualizer">
            <h4 className="cv-title">Waveform Comparison</h4>

            <div className="cv-track">
                <span className="cv-label">You</span>
                <canvas ref={userCanvasRef} width={600} height={80} className="cv-canvas" />
            </div>

            <div className="cv-track">
                <span className="cv-label">Reference</span>
                <canvas ref={refCanvasRef} width={600} height={80} className="cv-canvas" />
            </div>

            <div className="cv-legend">
                <span className="cv-legend-item"><Activity size={12} color="#10b981" /> Matches pattern</span>
                <span className="cv-legend-item"><Activity size={12} color="#2563eb" /> Reference pattern</span>
            </div>
        </div>
    );
}
