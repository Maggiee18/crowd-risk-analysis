import React, { useState } from 'react';
import { Play, Square, RefreshCw } from 'lucide-react';

const SCENARIOS = {
  normal: { label: 'Normal Flow', desc: 'Steady low-to-moderate density', color: 'emerald' },
  gradual_buildup: { label: 'Gradual Build-up', desc: 'Slowly increasing crowd', color: 'amber' },
  sudden_surge: { label: 'Sudden Surge', desc: 'Abrupt crowd spike', color: 'red' },
  panic_event: { label: 'Panic Event', desc: 'Normal → panic → return', color: 'red' },
  evacuation: { label: 'Evacuation', desc: 'Large crowd dispersing', color: 'purple' },
  concert: { label: 'Concert', desc: 'Waves of density', color: 'blue' },
};

export default function SimulationPanel({ onStart, onStop, isRunning, progress }) {
  const [selected, setSelected] = useState('normal');

  const colorMap = {
    emerald: 'border-emerald-500/20 hover:bg-emerald-500/5',
    amber: 'border-amber-500/20 hover:bg-amber-500/5',
    red: 'border-red-500/20 hover:bg-red-500/5',
    purple: 'border-purple-500/20 hover:bg-purple-500/5',
    blue: 'border-blue-500/20 hover:bg-blue-500/5',
  };

  return (
    <div className="glass-card">
      <div className="flex items-center gap-2 mb-4">
        <Play size={18} className="text-brand-400" />
        <h3 className="text-sm font-semibold text-white">Simulation Mode</h3>
        {isRunning && (
          <span className="ml-auto text-[10px] text-emerald-400 bg-emerald-500/15 px-2 py-0.5 rounded-full animate-pulse">
            RUNNING
          </span>
        )}
      </div>

      {/* Scenario grid */}
      <div className="grid grid-cols-2 gap-2 mb-4">
        {Object.entries(SCENARIOS).map(([key, s]) => (
          <button
            key={key}
            onClick={() => !isRunning && setSelected(key)}
            disabled={isRunning}
            className={`text-left p-3 rounded-xl border transition-all duration-200 ${
              selected === key && !isRunning
                ? 'bg-brand-600/15 border-brand-500/25'
                : `bg-surface-800/30 ${colorMap[s.color]} border-white/[0.04]`
            } ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}
          >
            <div className="text-xs font-semibold text-white mb-0.5">{s.label}</div>
            <div className="text-[10px] text-slate-500">{s.desc}</div>
          </button>
        ))}
      </div>

      {/* Progress bar */}
      {isRunning && progress > 0 && (
        <div className="mb-4">
          <div className="flex justify-between text-[10px] text-slate-500 mb-1">
            <span>Progress</span>
            <span>{(progress * 100).toFixed(0)}%</span>
          </div>
          <div className="h-1.5 bg-surface-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-600 to-brand-400 rounded-full transition-all duration-500"
              style={{ width: `${progress * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Controls */}
      <div className="flex gap-2">
        {!isRunning ? (
          <button onClick={() => onStart(selected)} className="btn-primary flex-1">
            <Play size={16} /> Start Simulation
          </button>
        ) : (
          <button onClick={onStop} className="btn-primary flex-1 !bg-red-600 hover:!bg-red-500 !shadow-red-600/25">
            <Square size={16} /> Stop
          </button>
        )}
      </div>
    </div>
  );
}
