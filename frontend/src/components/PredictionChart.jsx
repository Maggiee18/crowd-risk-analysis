import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine
} from 'recharts';
import { Brain } from 'lucide-react';

export default function PredictionChart({ predictions = {}, history = [] }) {
  const preds = predictions.predictions || [];
  const method = predictions.method || 'unknown';
  const confidence = predictions.confidence || 0;

  // Build chart data: last N history points + prediction points
  const recentHistory = history.slice(-20);

  const chartData = [
    ...recentHistory.map((h, i) => ({
      step: `H-${recentHistory.length - i}`,
      actual: h.people_count || 0,
      predicted: null,
      type: 'history',
    })),
    ...preds.map((p, i) => ({
      step: `+${i + 1}`,
      actual: null,
      predicted: p,
      type: 'prediction',
    })),
  ];

  // Add a bridge point
  if (recentHistory.length > 0 && preds.length > 0) {
    const lastActual = recentHistory[recentHistory.length - 1].people_count || 0;
    chartData.splice(recentHistory.length, 0, {
      step: 'Now',
      actual: lastActual,
      predicted: lastActual,
      type: 'bridge',
    });
  }

  return (
    <div className="glass-card">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain size={18} className="text-purple-400" />
          <h3 className="text-sm font-semibold t-heading">Crowd Prediction</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-[10px] t-muted uppercase">{method}</span>
          {confidence > 0 && (
            <span className="text-[10px] text-purple-400 bg-purple-500/15 px-2 py-0.5 rounded-full">
              {(confidence * 100).toFixed(0)}% conf
            </span>
          )}
        </div>
      </div>

      {chartData.length <= 1 ? (
        <div className="h-[220px] flex flex-col items-center justify-center t-muted text-sm">
          <Brain size={32} className="mb-2 opacity-30" />
          <p>Collecting data for prediction…</p>
          {predictions.steps_needed && (
            <p className="text-xs mt-1">{predictions.steps_needed} more samples needed</p>
          )}
        </div>
      ) : (
        <div className="h-[220px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="step"
                tick={{ fill: '#64748b', fontSize: 9 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                tickLine={false}
              />
              <YAxis
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={false}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1e293b',
                  border: '1px solid rgba(255,255,255,0.08)',
                  borderRadius: '10px',
                  fontSize: '11px',
                }}
              />
              <ReferenceLine x="Now" stroke="rgba(168,85,247,0.4)" strokeDasharray="4 4" label="" />
              <Line
                type="monotone"
                dataKey="actual"
                stroke="#6366f1"
                strokeWidth={2}
                dot={false}
                name="Actual"
                connectNulls={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                stroke="#a78bfa"
                strokeWidth={2}
                strokeDasharray="6 3"
                dot={{ r: 3, fill: '#a78bfa' }}
                name="Predicted"
                connectNulls={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {preds.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {preds.slice(0, 5).map((p, i) => (
            <span key={i} className="text-[10px] bg-purple-500/10 text-purple-300 px-2 py-0.5 rounded-full border border-purple-500/10">
              +{i + 1}: {p}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
