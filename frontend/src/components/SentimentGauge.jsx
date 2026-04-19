import React, { useMemo } from 'react';
import { Smile, Frown, Meh, AlertTriangle, Zap, Heart } from 'lucide-react';

/**
 * Feature #11: Crowd Sentiment Estimation
 * Estimates crowd "mood" from speed, density, flow, and anomaly data.
 */
export default function SentimentGauge({ features = {}, anomaly = {}, peopleCount = 0 }) {
  const sentiment = useMemo(() => {
    const speed = features.avg_magnitude || 0;
    const speedVar = features.speed_variance || 0;
    const density = features.density_per_frame || 0;
    const uniformity = features.direction_uniformity || 0;
    const isAnomaly = anomaly.is_anomaly || false;

    // Scoring system: 0 (calm) → 100 (panicking)
    let score = 0;

    // Speed factor
    if (speed > 20) score += 35;        // Running = panic
    else if (speed > 10) score += 20;   // Fast walking = agitated
    else if (speed < 1 && peopleCount > 5) score += 25; // Frozen = congestion stress
    else score += speed * 1.5;

    // Speed variance = erratic behavior
    score += Math.min(20, speedVar * 3);

    // Density stress
    score += Math.min(25, density * 200);

    // Counter-flow = confusion
    if (uniformity < 0.3 && peopleCount > 3) score += 15;

    // Anomaly = instant concern
    if (isAnomaly) score += 20;

    score = Math.max(0, Math.min(100, score));

    // Map score to mood
    if (score < 15) return { level: 'calm', label: 'Calm', emoji: '😊', color: '#10b981', icon: Smile, score, desc: 'Crowd is relaxed and moving normally' };
    if (score < 35) return { level: 'normal', label: 'Normal', emoji: '🙂', color: '#6366f1', icon: Smile, score, desc: 'Typical crowd behavior, no concerns' };
    if (score < 55) return { level: 'restless', label: 'Restless', emoji: '😐', color: '#f59e0b', icon: Meh, score, desc: 'Increased movement and density detected' };
    if (score < 75) return { level: 'agitated', label: 'Agitated', emoji: '😟', color: '#f97316', icon: Frown, score, desc: 'Erratic movement patterns, possible distress' };
    return { level: 'panicking', label: 'Panicking', emoji: '😱', color: '#ef4444', icon: AlertTriangle, score, desc: 'Critical — rapid/chaotic movement detected' };
  }, [features, anomaly, peopleCount]);

  const Icon = sentiment.icon;

  // Animated ring
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (circumference * sentiment.score) / 100;

  return (
    <div className="glass-card text-center" id="sentiment-gauge">
      <div className="flex items-center justify-center gap-2 mb-3">
        <Heart size={14} className="text-pink-400" />
        <h3 className="text-xs font-semibold t-muted uppercase tracking-wider">Crowd Sentiment</h3>
      </div>

      {/* Circular gauge */}
      <div className="relative w-28 h-28 mx-auto mb-3">
        <svg viewBox="0 0 100 100" className="w-full h-full -rotate-90">
          {/* Background circle */}
          <circle
            cx="50" cy="50" r={radius}
            fill="none"
            stroke="var(--panel-border)"
            strokeWidth="6"
          />
          {/* Value circle */}
          <circle
            cx="50" cy="50" r={radius}
            fill="none"
            stroke={sentiment.color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            className="transition-all duration-1000 ease-out"
            style={{ filter: `drop-shadow(0 0 6px ${sentiment.color}60)` }}
          />
        </svg>
        {/* Center content */}
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-3xl">{sentiment.emoji}</span>
          <span className="text-[10px] font-bold mt-0.5" style={{ color: sentiment.color }}>
            {sentiment.score.toFixed(0)}
          </span>
        </div>
      </div>

      {/* Label */}
      <div
        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold"
        style={{ background: sentiment.color + '20', color: sentiment.color, border: `1px solid ${sentiment.color}30` }}
      >
        <Icon size={12} />
        {sentiment.label}
      </div>

      <p className="text-[10px] t-muted mt-2 px-4">{sentiment.desc}</p>

      {/* Mood scale */}
      <div className="flex items-center gap-0.5 mt-3 px-2">
        {['😊', '🙂', '😐', '😟', '😱'].map((e, i) => (
          <div key={i} className="flex-1 text-center">
            <div
              className="h-1 rounded-full mb-1"
              style={{
                background: i <= Math.floor(sentiment.score / 25) ? sentiment.color : 'var(--panel-border)',
                opacity: i <= Math.floor(sentiment.score / 25) ? 1 : 0.3,
              }}
            />
            <span className="text-[10px]" style={{ opacity: i === Math.floor(sentiment.score / 25) ? 1 : 0.3 }}>{e}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
