import React from 'react';
import { AlertTriangle, XCircle, Info, Bell, Clock } from 'lucide-react';

export default function AlertPanel({ alerts = [], alertHistory = [] }) {
  const allAlerts = alerts.length > 0 ? alerts : [];

  const getIcon = (severity) => {
    switch (severity) {
      case 'critical': return <XCircle size={18} />;
      case 'warning': return <AlertTriangle size={18} />;
      default: return <Info size={18} />;
    }
  };

  const formatTime = (ts) => {
    try {
      if (typeof ts === 'number') return new Date(ts * 1000).toLocaleTimeString();
      return new Date(ts).toLocaleTimeString();
    } catch { return '—'; }
  };

  return (
    <div className="glass-card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bell size={18} className="text-amber-400" />
          <h3 className="text-sm font-semibold text-white">Alerts</h3>
        </div>
        {allAlerts.length > 0 && (
          <span className="text-[10px] font-bold text-red-400 bg-red-500/15 px-2 py-0.5 rounded-full">
            {allAlerts.length} ACTIVE
          </span>
        )}
      </div>

      {allAlerts.length === 0 ? (
        <div className="flex flex-col items-center py-6 text-slate-500">
          <div className="w-12 h-12 rounded-full bg-emerald-500/10 flex items-center justify-center mb-2">
            <Bell size={20} className="text-emerald-400" />
          </div>
          <p className="text-sm font-medium text-emerald-400">All Clear</p>
          <p className="text-xs mt-1">No active alerts</p>
        </div>
      ) : (
        <div className="space-y-2 max-h-[300px] overflow-y-auto">
          {allAlerts.map((alert, i) => (
            <div key={i} className={`alert-card ${alert.severity || 'warning'}`}>
              <div className="flex-shrink-0 mt-0.5">
                {getIcon(alert.severity)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="text-xs font-bold uppercase tracking-wider mb-0.5">
                  {(alert.type || '').replace(/_/g, ' ')}
                </div>
                <p className="text-xs opacity-80 leading-relaxed">{alert.message}</p>
                <div className="flex items-center gap-1 mt-1.5 text-[10px] opacity-50">
                  <Clock size={10} />
                  {formatTime(alert.timestamp)}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Recent history */}
      {alertHistory.length > 0 && (
        <div className="mt-4 pt-3 border-t border-white/[0.04]">
          <p className="text-[10px] text-slate-500 uppercase tracking-wider mb-2">Recent History</p>
          <div className="space-y-1.5 max-h-[120px] overflow-y-auto">
            {alertHistory.slice(-5).reverse().map((h, i) => (
              <div key={i} className="flex items-center gap-2 text-[11px] text-slate-500">
                <span className={`w-1.5 h-1.5 rounded-full ${h.severity === 'critical' ? 'bg-red-400' : 'bg-amber-400'}`} />
                <span className="truncate flex-1">{(h.type || '').replace(/_/g, ' ')}</span>
                <span>{formatTime(h.timestamp)}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
