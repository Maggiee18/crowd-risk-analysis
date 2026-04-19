import React, { useMemo } from 'react';
import { Brain, AlertTriangle, TrendingUp, TrendingDown, Zap, Users, ArrowRight } from 'lucide-react';

/**
 * Feature #5: AI Risk Explainability
 * Shows WHY the AI thinks the risk level is what it is — derived from features.
 */
export default function RiskExplainer({ features = {}, riskPrediction = {}, anomaly = {}, peopleCount = 0, history = [] }) {
  const risk = (riskPrediction.risk_level || 'low').toLowerCase();
  const conf = riskPrediction.confidence || 0;

  const factors = useMemo(() => {
    const f = [];
    const density = features.density_per_frame || 0;
    const speed = features.avg_magnitude || 0;
    const speedVar = features.speed_variance || 0;
    const growth = features.growth_rate || 0;
    const uniformity = features.direction_uniformity || 0;

    // Density factor
    if (density > 0.05) {
      f.push({
        icon: Users,
        label: 'High crowd density',
        detail: `${density.toFixed(3)} ppl/m² — ${density > 0.1 ? 'dangerously packed' : 'above normal'}`,
        severity: density > 0.1 ? 'high' : 'medium',
        contribution: Math.min(35, density * 300),
      });
    } else if (density > 0) {
      f.push({
        icon: Users,
        label: 'Normal crowd density',
        detail: `${density.toFixed(3)} ppl/m² — within safe limits`,
        severity: 'low',
        contribution: 5,
      });
    }

    // People count factor
    if (peopleCount > 25) {
      f.push({
        icon: Users,
        label: 'Large crowd detected',
        detail: `${peopleCount} people — exceeds threshold of 25`,
        severity: 'high',
        contribution: 25,
      });
    } else if (peopleCount > 12) {
      f.push({
        icon: Users,
        label: 'Moderate crowd',
        detail: `${peopleCount} people — approaching threshold`,
        severity: 'medium',
        contribution: 15,
      });
    }

    // Speed factor
    if (speed < 1 && peopleCount > 5) {
      f.push({
        icon: TrendingDown,
        label: 'Crowd congestion',
        detail: `Average speed ${speed.toFixed(1)} px/s — movement is stalled`,
        severity: 'high',
        contribution: 20,
      });
    } else if (speed > 15) {
      f.push({
        icon: Zap,
        label: 'Rapid movement',
        detail: `Average speed ${speed.toFixed(1)} px/s — possible panic/running`,
        severity: 'high',
        contribution: 25,
      });
    }

    // Speed variance
    if (speedVar > 5) {
      f.push({
        icon: Zap,
        label: 'Erratic movement',
        detail: `Speed variance ${speedVar.toFixed(1)} — inconsistent crowd behavior`,
        severity: 'medium',
        contribution: 15,
      });
    }

    // Growth rate
    if (growth > 10) {
      f.push({
        icon: TrendingUp,
        label: 'Crowd surging',
        detail: `${growth.toFixed(0)}% growth rate — rapid influx detected`,
        severity: 'high',
        contribution: 20,
      });
    } else if (growth < -10) {
      f.push({
        icon: TrendingDown,
        label: 'Crowd dispersing',
        detail: `${growth.toFixed(0)}% decline — possible evacuation`,
        severity: 'medium',
        contribution: 10,
      });
    }

    // Flow uniformity
    if (uniformity < 0.3 && peopleCount > 5) {
      f.push({
        icon: AlertTriangle,
        label: 'Counter-flow detected',
        detail: `Flow uniformity ${(uniformity * 100).toFixed(0)}% — groups moving in opposing directions`,
        severity: 'high',
        contribution: 20,
      });
    }

    // Anomaly
    if (anomaly.is_anomaly) {
      f.push({
        icon: AlertTriangle,
        label: 'Anomalous behavior',
        detail: `Score: ${(anomaly.anomaly_score || 0).toFixed(2)} — pattern deviates from normal`,
        severity: 'high',
        contribution: 30,
      });
    }

    // Historical pattern
    if (history.length > 10) {
      const recentHigh = history.slice(-10).filter(h => h.risk_level === 'high').length;
      if (recentHigh >= 5) {
        f.push({
          icon: Brain,
          label: 'Sustained high risk',
          detail: `${recentHigh}/10 recent frames were high risk — persistent concern`,
          severity: 'high',
          contribution: 15,
        });
      }
    }

    return f.sort((a, b) => b.contribution - a.contribution);
  }, [features, anomaly, peopleCount, history]);

  const severityColors = {
    low: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/15',
    medium: 'text-amber-400 bg-amber-500/10 border-amber-500/15',
    high: 'text-red-400 bg-red-500/10 border-red-500/15',
  };

  return (
    <div className="glass-card" id="risk-explainer">
      <div className="flex items-center gap-2 mb-4">
        <Brain size={16} className="text-violet-400" />
        <h3 className="text-sm font-semibold t-heading">AI Risk Explanation</h3>
        <span className={`ml-auto text-[10px] font-bold uppercase ${
          risk === 'high' ? 'text-red-400' : risk === 'medium' ? 'text-amber-400' : 'text-emerald-400'
        }`}>
          {risk} risk • {(conf * 100).toFixed(0)}%
        </span>
      </div>

      {factors.length === 0 ? (
        <div className="text-center py-4 t-muted text-xs">
          <Brain size={24} className="mx-auto mb-2 opacity-30" />
          No significant risk factors detected
        </div>
      ) : (
        <div className="space-y-2">
          {factors.map((f, i) => {
            const Icon = f.icon;
            return (
              <div key={i} className={`flex items-start gap-2.5 p-2.5 rounded-xl border ${severityColors[f.severity]}`}>
                <Icon size={14} className="mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold">{f.label}</span>
                    <div className="flex-1 h-1 rounded-full bg-black/20 overflow-hidden">
                      <div
                        className="h-full rounded-full transition-all duration-500"
                        style={{
                          width: `${f.contribution}%`,
                          background: f.severity === 'high' ? '#ef4444' : f.severity === 'medium' ? '#f59e0b' : '#10b981'
                        }}
                      />
                    </div>
                    <span className="text-[9px] opacity-60">{f.contribution.toFixed(0)}%</span>
                  </div>
                  <p className="text-[10px] opacity-70 mt-0.5">{f.detail}</p>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Summary sentence */}
      {factors.length > 0 && (
        <div className="mt-3 p-2.5 rounded-lg text-[11px] t-muted" style={{ background: 'var(--panel-bg)' }}>
          <strong>Summary:</strong> Risk is {risk} primarily due to{' '}
          {factors.slice(0, 3).map(f => f.label.toLowerCase()).join(', ')}
          {factors.length > 3 ? ` and ${factors.length - 3} other factor(s)` : ''}.
        </div>
      )}
    </div>
  );
}
