import React from 'react';
import { Map } from 'lucide-react';

export default function ZoneMap({ zones = [] }) {
  const getColor = (risk) => {
    switch (risk) {
      case 'high': return { bg: 'bg-red-500/20', border: 'border-red-500/30', text: 'text-red-400', glow: 'shadow-red-500/10' };
      case 'medium': return { bg: 'bg-amber-500/15', border: 'border-amber-500/25', text: 'text-amber-400', glow: 'shadow-amber-500/10' };
      default: return { bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', text: 'text-emerald-400', glow: '' };
    }
  };

  // 3x3 grid
  const grid = [];
  for (let r = 0; r < 3; r++) {
    const row = [];
    for (let c = 0; c < 3; c++) {
      const zone = zones.find(z => z.id === `zone_${r}_${c}`) || {
        label: `Zone ${r * 3 + c + 1}`,
        people_count: 0,
        risk: 'low',
      };
      row.push(zone);
    }
    grid.push(row);
  }

  return (
    <div className="glass-card">
      <div className="flex items-center gap-2 mb-4">
        <Map size={18} className="text-cyan-400" />
        <h3 className="text-sm font-semibold text-white">Zone Analysis</h3>
      </div>

      <div className="grid grid-cols-3 gap-2">
        {grid.flat().map((zone, i) => {
          const colors = getColor(zone.risk);
          return (
            <div
              key={i}
              className={`zone-cell ${zone.risk} p-3 shadow-lg ${colors.glow} transition-all duration-500`}
            >
              <span className="text-[10px] text-slate-500 mb-1">{zone.label}</span>
              <span className={`text-lg font-bold ${colors.text}`}>{zone.people_count || 0}</span>
              <span className={`text-[9px] uppercase font-semibold mt-0.5 ${colors.text}`}>
                {zone.risk}
              </span>
            </div>
          );
        })}
      </div>

      {/* Legend */}
      <div className="flex items-center justify-center gap-4 mt-4 pt-3 border-t border-white/[0.04]">
        {['low', 'medium', 'high'].map(level => (
          <div key={level} className="flex items-center gap-1.5">
            <span className={`w-2 h-2 rounded-full ${level === 'low' ? 'bg-emerald-400' : level === 'medium' ? 'bg-amber-400' : 'bg-red-400'}`} />
            <span className="text-[10px] text-slate-500 capitalize">{level}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
