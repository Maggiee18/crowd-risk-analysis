import React, { useState, useCallback } from 'react';
import {
  AlertOctagon, Radio, Shield, Download, Clock, X, PhoneCall
} from 'lucide-react';

/**
 * Feature #10: Emergency Broadcast Integration
 * One-click emergency mode with incident freezing and report generation.
 */
export default function EmergencyPanel({ currentState, history, alerts, onEmergencyToggle }) {
  const [emergencyActive, setEmergencyActive] = useState(false);
  const [frozenState, setFrozenState] = useState(null);
  const [emergencyLog, setEmergencyLog] = useState([]);
  const [showPanel, setShowPanel] = useState(false);

  const activateEmergency = useCallback(() => {
    const now = new Date();

    // Freeze current state as evidence
    const frozen = {
      timestamp: now.toISOString(),
      frame: currentState?.current_frame,
      people_count: currentState?.people_count || 0,
      risk: currentState?.risk_prediction || {},
      features: currentState?.features || {},
      anomaly: currentState?.anomaly_detection || {},
      alerts: [...(alerts || [])],
    };

    setFrozenState(frozen);
    setEmergencyActive(true);
    setEmergencyLog(prev => [...prev, {
      action: 'EMERGENCY ACTIVATED',
      time: now.toLocaleTimeString(),
      detail: `People: ${frozen.people_count}, Risk: ${frozen.risk.risk_level}`,
    }]);
    onEmergencyToggle?.(true);
  }, [currentState, alerts, onEmergencyToggle]);

  const deactivateEmergency = useCallback(() => {
    setEmergencyActive(false);
    setEmergencyLog(prev => [...prev, {
      action: 'EMERGENCY DEACTIVATED',
      time: new Date().toLocaleTimeString(),
      detail: 'Situation resolved',
    }]);
    onEmergencyToggle?.(false);
  }, [onEmergencyToggle]);

  const downloadEvidence = useCallback(() => {
    if (!frozenState) return;
    const data = {
      emergency_report: true,
      frozen_at: frozenState.timestamp,
      people_count: frozenState.people_count,
      risk_level: frozenState.risk.risk_level,
      risk_confidence: frozenState.risk.confidence,
      features: frozenState.features,
      anomaly: frozenState.anomaly,
      alerts: frozenState.alerts,
      emergency_log: emergencyLog,
    };

    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `emergency_evidence_${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [frozenState, emergencyLog]);

  return (
    <>
      {/* Emergency trigger button — always visible in header */}
      <button
        onClick={() => setShowPanel(true)}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-semibold transition-all ${
          emergencyActive
            ? 'bg-red-600 text-white animate-pulse shadow-lg shadow-red-600/30'
            : 'bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20'
        }`}
        id="emergency-btn"
      >
        <AlertOctagon size={14} />
        {emergencyActive ? 'EMERGENCY ACTIVE' : 'Emergency'}
      </button>

      {/* Emergency panel modal */}
      {showPanel && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm animate-fade-in">
          <div className={`glass-card w-full max-w-lg mx-4 space-y-4 ${emergencyActive ? 'ring-2 ring-red-500 ring-offset-2 ring-offset-black' : ''}`}>
            {/* Header */}
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <AlertOctagon size={20} className="text-red-400" />
                <h3 className="text-base font-bold t-heading">Emergency Control</h3>
              </div>
              <button onClick={() => setShowPanel(false)} className="t-muted hover:text-red-400">
                <X size={16} />
              </button>
            </div>

            {/* Status */}
            <div className={`p-4 rounded-xl text-center ${
              emergencyActive
                ? 'bg-red-500/15 border-2 border-red-500/30'
                : 'border border-dashed'
            }`} style={{ borderColor: emergencyActive ? undefined : 'var(--card-border)' }}>
              {emergencyActive ? (
                <>
                  <div className="w-12 h-12 mx-auto mb-3 rounded-full bg-red-500/20 flex items-center justify-center animate-pulse">
                    <Radio size={24} className="text-red-400" />
                  </div>
                  <p className="text-red-400 font-bold text-sm">EMERGENCY MODE ACTIVE</p>
                  <p className="text-[11px] t-muted mt-1">Scene frozen at {frozenState?.timestamp ? new Date(frozenState.timestamp).toLocaleTimeString() : '--'}</p>
                  <p className="text-xs t-muted mt-1">People: {frozenState?.people_count} • Risk: {frozenState?.risk?.risk_level?.toUpperCase()}</p>
                </>
              ) : (
                <>
                  <Shield size={32} className="mx-auto mb-2 t-muted opacity-30" />
                  <p className="text-xs t-muted">No active emergency</p>
                </>
              )}
            </div>

            {/* Action buttons */}
            <div className="grid grid-cols-2 gap-2">
              {!emergencyActive ? (
                <button
                  onClick={activateEmergency}
                  className="col-span-2 py-3 rounded-xl bg-red-600 hover:bg-red-500 text-white font-bold text-sm flex items-center justify-center gap-2 transition-all hover:scale-[1.02] active:scale-[0.98]"
                >
                  <AlertOctagon size={18} /> ACTIVATE EMERGENCY
                </button>
              ) : (
                <>
                  <button
                    onClick={deactivateEmergency}
                    className="py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex items-center justify-center gap-2"
                  >
                    <Shield size={14} /> Resolve
                  </button>
                  <button
                    onClick={downloadEvidence}
                    className="py-2.5 rounded-xl bg-blue-600 hover:bg-blue-500 text-white font-semibold text-xs flex items-center justify-center gap-2"
                  >
                    <Download size={14} /> Save Evidence
                  </button>
                </>
              )}
            </div>

            {/* Simulated actions */}
            {emergencyActive && (
              <div className="space-y-1.5">
                <p className="text-[10px] t-muted uppercase font-semibold tracking-wider">Triggered Actions:</p>
                {[
                  { icon: PhoneCall, label: 'Security team notified', status: 'Sent' },
                  { icon: Radio, label: 'PA system alert broadcast', status: 'Active' },
                  { icon: Download, label: 'Evidence auto-saved', status: 'Done' },
                ].map((action, i) => (
                  <div key={i} className="flex items-center gap-2 px-3 py-2 rounded-lg text-xs" style={{ background: 'var(--panel-bg)' }}>
                    <action.icon size={12} className="text-red-400" />
                    <span className="t-body flex-1">{action.label}</span>
                    <span className="text-[10px] text-emerald-400 font-semibold">{action.status}</span>
                  </div>
                ))}
              </div>
            )}

            {/* Emergency log */}
            {emergencyLog.length > 0 && (
              <div className="space-y-1">
                <p className="text-[10px] t-muted uppercase font-semibold tracking-wider">Activity Log:</p>
                <div className="max-h-[120px] overflow-y-auto space-y-1">
                  {emergencyLog.map((log, i) => (
                    <div key={i} className="flex items-center gap-2 text-[10px] px-2 py-1.5 rounded-lg" style={{ background: 'var(--panel-bg)' }}>
                      <Clock size={10} className="t-muted shrink-0" />
                      <span className="t-muted tabular-nums">{log.time}</span>
                      <span className="font-semibold text-red-400">{log.action}</span>
                      <span className="t-muted truncate">{log.detail}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
