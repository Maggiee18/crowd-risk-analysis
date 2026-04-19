import React, { useMemo } from 'react';
import { Clock, AlertTriangle, Users, TrendingUp } from 'lucide-react';

/**
 * Feature #4: Session Timeline / Playback
 * Shows a timeline of risk events during the current session.
 */
export default function SessionTimeline({ history = [], alerts = [], onJumpTo }) {
  // Build timeline events from history
  const events = useMemo(() => {
    const evts = [];
    let prevRisk = 'low';

    history.forEach((h, i) => {
      const risk = (h.risk_level || 'low').toLowerCase();
      const time = h.timestamp ? new Date(h.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) : `T${i}`;

      // Risk level change
      if (risk !== prevRisk) {
        evts.push({
          type: risk === 'high' ? 'danger' : risk === 'medium' ? 'warning' : 'clear',
          time,
          index: i,
          label: risk === 'high' ? 'Risk → HIGH' : risk === 'medium' ? 'Risk → MEDIUM' : 'Risk → LOW',
          count: h.people_count || 0,
        });
        prevRisk = risk;
      }

      // People count spike
      if (i > 0 && h.people_count > (history[i - 1]?.people_count || 0) * 1.5 && h.people_count > 5) {
        evts.push({
          type: 'spike',
          time,
          index: i,
          label: `Crowd spike: ${h.people_count} people`,
          count: h.people_count,
        });
      }
    });

    return evts.slice(-20); // Last 20 events
  }, [history]);

  // Current risk color for timeline bar
  const riskSegments = useMemo(() => {
    if (history.length === 0) return [];
    const segments = [];
    let start = 0;
    let currentRisk = history[0]?.risk_level || 'low';

    for (let i = 1; i <= history.length; i++) {
      const risk = history[i]?.risk_level || currentRisk;
      if (risk !== currentRisk || i === history.length) {
        const pct = ((i - start) / history.length) * 100;
        segments.push({ risk: currentRisk, width: pct });
        start = i;
        currentRisk = risk;
      }
    }
    return segments;
  }, [history]);

  const riskColors = {
    low: '#10b981',
    medium: '#f59e0b',
    high: '#ef4444',
  };

  const eventIcons = {
    danger: AlertTriangle,
    warning: AlertTriangle,
    clear: TrendingUp,
    spike: Users,
  };

  const eventColors = {
    danger: 'text-red-400 bg-red-500/15 border-red-500/20',
    warning: 'text-amber-400 bg-amber-500/15 border-amber-500/20',
    clear: 'text-emerald-400 bg-emerald-500/15 border-emerald-500/20',
    spike: 'text-blue-400 bg-blue-500/15 border-blue-500/20',
  };

  return (
    <div className="glass-card" id="session-timeline">
      <div className="flex items-center gap-2 mb-4">
        <Clock size={16} className="text-cyan-400" />
        <h3 className="text-sm font-semibold t-heading">Session Timeline</h3>
        <span className="ml-auto text-[10px] t-muted">{events.length} events • {history.length} frames</span>
      </div>

      {/* Risk color bar */}
      <div className="flex h-2 rounded-full overflow-hidden mb-4" style={{ background: 'var(--panel-bg)' }}>
        {riskSegments.map((seg, i) => (
          <div
            key={i}
            style={{
              width: `${seg.width}%`,
              backgroundColor: riskColors[seg.risk] || '#64748b',
              transition: 'width 0.3s ease',
            }}
          />
        ))}
      </div>

      {/* Event list */}
      {events.length === 0 ? (
        <div className="text-center py-6 t-muted text-xs">
          <Clock size={24} className="mx-auto mb-2 opacity-30" />
          No events yet — monitoring…
        </div>
      ) : (
        <div className="space-y-1.5 max-h-[200px] overflow-y-auto pr-1">
          {events.map((evt, i) => {
            const Icon = eventIcons[evt.type] || Clock;
            return (
              <button
                key={i}
                onClick={() => onJumpTo?.(evt.index)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg border text-xs transition-all hover:scale-[1.01] ${eventColors[evt.type]}`}
              >
                <Icon size={12} />
                <span className="tabular-nums font-mono text-[10px] opacity-70">{evt.time}</span>
                <span className="font-medium flex-1 text-left">{evt.label}</span>
                <span className="text-[10px] opacity-60">{evt.count} ppl</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Time markers */}
      {history.length > 0 && (
        <div className="flex justify-between mt-3 text-[9px] t-muted">
          <span>{history[0]?.timestamp ? new Date(history[0].timestamp).toLocaleTimeString() : 'Start'}</span>
          <span>{history[history.length - 1]?.timestamp ? new Date(history[history.length - 1].timestamp).toLocaleTimeString() : 'Now'}</span>
        </div>
      )}
    </div>
  );
}
