import React from 'react';
import { Shield, ArrowRight, Zap } from 'lucide-react';

export default function ControlPanel({ suggestions = [] }) {
  const getActionIcon = (action) => {
    if (action.includes('redirect')) return '↗️';
    if (action.includes('close')) return '🚫';
    if (action.includes('open')) return '🚪';
    if (action.includes('deploy')) return '👮';
    if (action.includes('announce')) return '📢';
    if (action.includes('barrier')) return '🚧';
    if (action === 'no_action') return '✅';
    return '⚡';
  };

  const getConfidenceColor = (conf) => {
    if (conf >= 0.7) return 'text-emerald-400 bg-emerald-500/15';
    if (conf >= 0.4) return 'text-amber-400 bg-amber-500/15';
    return 'text-slate-400 bg-slate-500/15';
  };

  return (
    <div className="glass-card">
      <div className="flex items-center gap-2 mb-4">
        <Shield size={18} className="text-cyan-400" />
        <h3 className="text-sm font-semibold text-white">Control Suggestions</h3>
        <Zap size={12} className="text-amber-400 ml-1" />
        <span className="text-[10px] text-slate-500">AI-powered</span>
      </div>

      {suggestions.length === 0 ? (
        <div className="flex flex-col items-center py-6 text-slate-500">
          <Shield size={28} className="mb-2 opacity-30" />
          <p className="text-sm">No suggestions yet</p>
          <p className="text-xs mt-1">Waiting for more data to analyze</p>
        </div>
      ) : (
        <div className="space-y-2.5">
          {suggestions.map((s, i) => (
            <div
              key={i}
              className={`flex items-start gap-3 p-3 rounded-xl border transition-all duration-300 ${
                i === 0
                  ? 'bg-brand-600/10 border-brand-500/15'
                  : 'bg-surface-800/40 border-white/[0.04] opacity-80'
              }`}
            >
              <span className="text-lg flex-shrink-0">{getActionIcon(s.action)}</span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-xs font-bold text-white">
                    #{s.priority}
                  </span>
                  <span className="text-xs text-slate-300 font-medium">
                    {(s.action || '').replace(/_/g, ' ').toUpperCase()}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  {s.description}
                </p>
              </div>
              <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold flex-shrink-0 ${getConfidenceColor(s.confidence)}`}>
                {(s.confidence * 100).toFixed(0)}%
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
