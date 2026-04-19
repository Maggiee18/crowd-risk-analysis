import React from 'react';

export default function RiskGauge({ riskLevel, confidence }) {
  const level = (riskLevel || 'low').toLowerCase();

  const config = {
    low: { color: '#10b981', bg: 'from-emerald-500/20 to-emerald-900/10', label: 'LOW', angle: 45 },
    medium: { color: '#f59e0b', bg: 'from-amber-500/20 to-amber-900/10', label: 'MEDIUM', angle: 90 },
    high: { color: '#ef4444', bg: 'from-red-500/20 to-red-900/10', label: 'HIGH', angle: 150 },
  };
  const c = config[level] || config.low;
  const pct = (confidence || 0) * 100;
  const visualPct = level === 'high' ? 85 : level === 'medium' ? 50 : 20;
  const radius = 56;
  const circumference = Math.PI * radius; // semicircle
  const dashOffset = circumference - (circumference * visualPct) / 100;

  return (
    <div className={`glass-card bg-gradient-to-br ${c.bg} text-center`}>
      <h3 className="text-xs font-semibold t-muted uppercase tracking-wider mb-3">Risk Level</h3>

      <div className="relative w-36 h-20 mx-auto mb-2">
        <svg viewBox="0 0 120 68" className="w-full h-full">
          {/* Background arc */}
          <path
            d="M 10 60 A 50 50 0 0 1 110 60"
            fill="none"
            stroke="rgba(255,255,255,0.06)"
            strokeWidth="10"
            strokeLinecap="round"
          />
          {/* Value arc */}
          <path
            d="M 10 60 A 50 50 0 0 1 110 60"
            fill="none"
            stroke={c.color}
            strokeWidth="10"
            strokeLinecap="round"
            strokeDasharray={`${circumference}`}
            strokeDashoffset={dashOffset}
            className="transition-all duration-1000 ease-out"
            style={{ filter: `drop-shadow(0 0 6px ${c.color}60)` }}
          />
          {/* Center text */}
          <text x="60" y="52" textAnchor="middle" fill={c.color} fontSize="14" fontWeight="800" fontFamily="Inter">
            {c.label}
          </text>
        </svg>
      </div>

      <div className={`risk-badge ${level} mx-auto`}>
        {level === 'high' && <span className="w-1.5 h-1.5 rounded-full bg-red-400 animate-pulse mr-1" />}
        {c.label} RISK
      </div>
      <p className="text-[11px] t-muted mt-2">Confidence: {pct.toFixed(1)}%</p>
    </div>
  );
}
