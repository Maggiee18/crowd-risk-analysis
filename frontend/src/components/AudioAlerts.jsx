import { useEffect, useRef, useCallback } from 'react';

/**
 * Feature #2: Audio Alert System
 * Plays distinct tones for different risk levels using Web Audio API.
 * No external audio files needed — generates tones procedurally.
 */
export default function AudioAlerts({ riskLevel, anomalyDetected, enabled = true }) {
  const audioCtxRef = useRef(null);
  const lastRiskRef = useRef('low');
  const lastAlertTime = useRef(0);

  const getAudioCtx = useCallback(() => {
    if (!audioCtxRef.current) {
      audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
    }
    return audioCtxRef.current;
  }, []);

  // Play a tone with given frequency, duration, and wave type
  const playTone = useCallback((freq, duration, type = 'sine', volume = 0.3) => {
    try {
      const ctx = getAudioCtx();
      if (ctx.state === 'suspended') ctx.resume();

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = type;
      osc.frequency.setValueAtTime(freq, ctx.currentTime);

      gain.gain.setValueAtTime(volume, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + duration);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + duration);
    } catch (e) {
      // Audio not supported or blocked
    }
  }, [getAudioCtx]);

  // Play escalating alert pattern
  const playMediumAlert = useCallback(() => {
    playTone(440, 0.3, 'sine', 0.2);           // A4
    setTimeout(() => playTone(554, 0.3, 'sine', 0.2), 350);  // C#5
  }, [playTone]);

  const playHighAlert = useCallback(() => {
    playTone(880, 0.15, 'sawtooth', 0.15);
    setTimeout(() => playTone(880, 0.15, 'sawtooth', 0.15), 200);
    setTimeout(() => playTone(1100, 0.3, 'sawtooth', 0.15), 400);
  }, [playTone]);

  const playAnomalyAlert = useCallback(() => {
    playTone(220, 0.5, 'square', 0.1);
    setTimeout(() => playTone(330, 0.5, 'square', 0.1), 600);
    setTimeout(() => playTone(220, 0.5, 'square', 0.1), 1200);
  }, [playTone]);

  // React to risk level changes
  useEffect(() => {
    if (!enabled) return;

    const now = Date.now();
    if (now - lastAlertTime.current < 5000) return; // Debounce 5s

    const level = (riskLevel || 'low').toLowerCase();

    if (level !== lastRiskRef.current) {
      if (level === 'medium' && lastRiskRef.current === 'low') {
        playMediumAlert();
        lastAlertTime.current = now;
      } else if (level === 'high') {
        playHighAlert();
        lastAlertTime.current = now;
      }
      lastRiskRef.current = level;
    }
  }, [riskLevel, enabled, playMediumAlert, playHighAlert]);

  // React to anomaly detection
  useEffect(() => {
    if (!enabled || !anomalyDetected) return;

    const now = Date.now();
    if (now - lastAlertTime.current < 8000) return; // Longer debounce for anomaly

    playAnomalyAlert();
    lastAlertTime.current = now;
  }, [anomalyDetected, enabled, playAnomalyAlert]);

  // Cleanup
  useEffect(() => {
    return () => {
      if (audioCtxRef.current) {
        audioCtxRef.current.close();
      }
    };
  }, []);

  return null; // Invisible component
}
