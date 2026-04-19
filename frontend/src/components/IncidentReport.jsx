import React, { useState, useCallback, useRef } from 'react';
import { Download, FileText, Camera, X, Loader2, CheckCircle } from 'lucide-react';

/**
 * Feature #1: Incident Screenshot & Report Export
 * Captures current state snapshot and generates a downloadable HTML report.
 */
export default function IncidentReport({ currentState, history, alerts, heatmapData, theme }) {
  const [showModal, setShowModal] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [generated, setGenerated] = useState(false);

  const generateReport = useCallback(() => {
    setGenerating(true);

    const now = new Date();
    const risk = currentState?.risk_prediction || {};
    const features = currentState?.features || {};
    const anomaly = currentState?.anomaly_detection || {};
    const count = currentState?.people_count || 0;
    const frameB64 = currentState?.current_frame;

    const riskColor = { low: '#10b981', medium: '#f59e0b', high: '#ef4444' }[risk.risk_level] || '#94a3b8';

    // Build HTML report
    const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>CrowdGuard Incident Report — ${now.toISOString()}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8fafc; color: #1e293b; padding: 40px; }
    .header { display: flex; align-items: center; gap: 16px; margin-bottom: 32px; padding-bottom: 20px; border-bottom: 2px solid #e2e8f0; }
    .logo { width: 48px; height: 48px; background: #4f46e5; border-radius: 12px; display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 20px; }
    .title { font-size: 24px; font-weight: 700; }
    .subtitle { font-size: 12px; color: #64748b; margin-top: 2px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }
    .card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; }
    .card h3 { font-size: 12px; text-transform: uppercase; color: #64748b; letter-spacing: 1px; margin-bottom: 8px; }
    .card .value { font-size: 28px; font-weight: 700; }
    .risk-badge { display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 600; font-size: 14px; color: white; }
    .frame-img { width: 100%; border-radius: 12px; border: 1px solid #e2e8f0; }
    .alert-item { padding: 10px 16px; border-left: 3px solid #ef4444; background: #fef2f2; border-radius: 0 8px 8px 0; margin-bottom: 8px; font-size: 13px; }
    .section { margin-bottom: 24px; }
    .section h2 { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #0f172a; }
    .meta { font-size: 11px; color: #94a3b8; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 12px; text-align: left; border-bottom: 1px solid #f1f5f9; font-size: 13px; }
    th { background: #f8fafc; font-weight: 600; color: #475569; }
    .footer { margin-top: 40px; padding-top: 20px; border-top: 1px solid #e2e8f0; text-align: center; font-size: 11px; color: #94a3b8; }
  </style>
</head>
<body>
  <div class="header">
    <div class="logo">CG</div>
    <div>
      <div class="title">CrowdGuard Incident Report</div>
      <div class="subtitle">Generated: ${now.toLocaleString()} | Report ID: ${Date.now().toString(36).toUpperCase()}</div>
    </div>
  </div>

  <div class="grid">
    <div class="card">
      <h3>People Detected</h3>
      <div class="value">${count}</div>
    </div>
    <div class="card">
      <h3>Risk Level</h3>
      <div class="value">
        <span class="risk-badge" style="background: ${riskColor}">
          ${(risk.risk_level || 'unknown').toUpperCase()}
        </span>
        <span style="font-size:14px; color:#64748b; margin-left:8px">${((risk.confidence || 0) * 100).toFixed(0)}% confidence</span>
      </div>
    </div>
    <div class="card">
      <h3>Crowd Density</h3>
      <div class="value">${(features.density_per_frame || 0).toFixed(3)} <span style="font-size:14px;color:#64748b">ppl/m²</span></div>
    </div>
    <div class="card">
      <h3>Anomaly Status</h3>
      <div class="value" style="color: ${anomaly.is_anomaly ? '#ef4444' : '#10b981'}">
        ${anomaly.is_anomaly ? '⚠️ DETECTED' : '✅ Normal'}
      </div>
    </div>
  </div>

  ${frameB64 ? `
  <div class="section">
    <h2>📸 Captured Frame</h2>
    <img src="data:image/jpeg;base64,${frameB64}" class="frame-img" alt="Incident frame" />
  </div>` : ''}

  <div class="section">
    <h2>📊 Feature Summary</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Average Speed</td><td>${(features.avg_magnitude || 0).toFixed(2)} px/s</td></tr>
      <tr><td>Speed Variance</td><td>${(features.speed_variance || 0).toFixed(3)}</td></tr>
      <tr><td>Density</td><td>${(features.density_per_frame || 0).toFixed(4)}</td></tr>
      <tr><td>Growth Rate</td><td>${(features.growth_rate || 0).toFixed(2)}%</td></tr>
      <tr><td>Dominant Direction</td><td>${(features.dominant_direction || 0).toFixed(1)}°</td></tr>
      <tr><td>Flow Uniformity</td><td>${(features.direction_uniformity || 0).toFixed(3)}</td></tr>
    </table>
  </div>

  ${alerts.length > 0 ? `
  <div class="section">
    <h2>🚨 Active Alerts (${alerts.length})</h2>
    ${alerts.map(a => `<div class="alert-item"><strong>${a.type || a.severity || 'Alert'}:</strong> ${a.message || a.description || JSON.stringify(a)}</div>`).join('')}
  </div>` : ''}

  <div class="section">
    <h2>📈 Recent History (last ${Math.min(history.length, 20)} points)</h2>
    <table>
      <tr><th>Time</th><th>Count</th><th>Density</th><th>Risk</th></tr>
      ${history.slice(-20).map(h => `
        <tr>
          <td>${h.timestamp ? new Date(h.timestamp).toLocaleTimeString() : '-'}</td>
          <td>${h.people_count || 0}</td>
          <td>${(h.density || 0).toFixed(3)}</td>
          <td>${h.risk_level || 'low'}</td>
        </tr>
      `).join('')}
    </table>
  </div>

  <div class="footer">
    CrowdGuard AI Risk System v2.0 • This report was automatically generated<br/>
    Contains ${history.length} historical data points • ${alerts.length} active alerts
  </div>
</body>
</html>`;

    // Download as HTML
    const blob = new Blob([html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `CrowdGuard_Report_${now.toISOString().replace(/[:.]/g, '-')}.html`;
    link.click();
    URL.revokeObjectURL(url);

    setGenerating(false);
    setGenerated(true);
    setTimeout(() => setGenerated(false), 3000);
  }, [currentState, history, alerts]);

  return (
    <>
      {/* Trigger button */}
      <button
        onClick={() => setShowModal(true)}
        className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-300"
        style={{
          background: 'var(--panel-bg)',
          border: '1px solid var(--card-border)',
          color: 'var(--color-text-muted)',
        }}
        title="Generate incident report"
        id="incident-report-btn"
      >
        <FileText size={14} />
        Report
      </button>

      {/* Modal */}
      {showModal && (
        <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/50 backdrop-blur-sm animate-fade-in">
          <div className="glass-card w-full max-w-md mx-4 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <FileText size={18} className="text-brand-400" />
                <h3 className="text-sm font-semibold t-heading">Generate Incident Report</h3>
              </div>
              <button onClick={() => setShowModal(false)} className="t-muted hover:text-red-400 transition-colors">
                <X size={16} />
              </button>
            </div>

            <div className="space-y-2 text-xs t-muted">
              <p>The report will include:</p>
              <ul className="list-disc pl-4 space-y-1">
                <li>Current frame screenshot with annotations</li>
                <li>People count, density & risk assessment</li>
                <li>Feature summary (speed, flow, anomalies)</li>
                <li>Active alerts ({alerts.length})</li>
                <li>Historical data ({history.length} points)</li>
              </ul>
            </div>

            <div className="flex gap-2">
              <button
                onClick={() => { generateReport(); setShowModal(false); }}
                disabled={generating}
                className="btn-primary flex-1"
              >
                {generating ? (
                  <><Loader2 size={14} className="animate-spin" /> Generating…</>
                ) : generated ? (
                  <><CheckCircle size={14} /> Downloaded!</>
                ) : (
                  <><Download size={14} /> Download Report</>
                )}
              </button>
              <button onClick={() => setShowModal(false)} className="btn-ghost">Cancel</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
