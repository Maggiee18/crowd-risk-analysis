import React, { useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Legend
} from 'recharts';
import { GitCompare, ArrowRight } from 'lucide-react';

/**
 * Feature #8: Comparative Analytics
 * Split view comparing two time periods from the session history.
 */
export default function ComparativeView({ history = [] }) {
  const halfIdx = Math.floor(history.length / 2);

  const analysis = useMemo(() => {
    if (history.length < 10) return null;

    const period1 = history.slice(0, halfIdx);
    const period2 = history.slice(halfIdx);

    const avg = (arr, key) => arr.reduce((s, h) => s + (h[key] || 0), 0) / (arr.length || 1);
    const max = (arr, key) => Math.max(...arr.map(h => h[key] || 0), 0);
    const riskCount = (arr, level) => arr.filter(h => h.risk_level === level).length;

    const p1Stats = {
      avgCount: avg(period1, 'people_count'),
      maxCount: max(period1, 'people_count'),
      avgDensity: avg(period1, 'density'),
      highRisk: riskCount(period1, 'high'),
      medRisk: riskCount(period1, 'medium'),
      frames: period1.length,
    };

    const p2Stats = {
      avgCount: avg(period2, 'people_count'),
      maxCount: max(period2, 'people_count'),
      avgDensity: avg(period2, 'density'),
      highRisk: riskCount(period2, 'high'),
      medRisk: riskCount(period2, 'medium'),
      frames: period2.length,
    };

    return { p1Stats, p2Stats, period1, period2 };
  }, [history, halfIdx]);

  // Merge for overlay chart
  const chartData = useMemo(() => {
    if (!analysis) return [];
    const maxLen = Math.max(analysis.period1.length, analysis.period2.length);
    const data = [];
    for (let i = 0; i < maxLen; i++) {
      data.push({
        index: i,
        'Period 1': analysis.period1[i]?.people_count || null,
        'Period 2': analysis.period2[i]?.people_count || null,
      });
    }
    return data;
  }, [analysis]);

  const delta = (v1, v2) => {
    if (v1 === 0) return v2 > 0 ? '+∞' : '0';
    const pct = ((v2 - v1) / v1 * 100);
    return pct >= 0 ? `+${pct.toFixed(0)}%` : `${pct.toFixed(0)}%`;
  };

  const deltaColor = (v1, v2, higherIsBad = true) => {
    if (v2 > v1) return higherIsBad ? 'text-red-400' : 'text-emerald-400';
    if (v2 < v1) return higherIsBad ? 'text-emerald-400' : 'text-red-400';
    return 't-muted';
  };

  if (!analysis) {
    return (
      <div className="glass-card" id="comparative-view">
        <div className="flex items-center gap-2 mb-4">
          <GitCompare size={16} className="text-teal-400" />
          <h3 className="text-sm font-semibold t-heading">Comparative Analytics</h3>
        </div>
        <div className="text-center py-8 t-muted text-xs">
          <GitCompare size={28} className="mx-auto mb-2 opacity-30" />
          Need at least 10 data points to compare
        </div>
      </div>
    );
  }

  const { p1Stats, p2Stats } = analysis;

  const metrics = [
    { label: 'Avg People', v1: p1Stats.avgCount, v2: p2Stats.avgCount, fmt: v => v.toFixed(1) },
    { label: 'Peak Count', v1: p1Stats.maxCount, v2: p2Stats.maxCount, fmt: v => v.toFixed(0) },
    { label: 'Avg Density', v1: p1Stats.avgDensity, v2: p2Stats.avgDensity, fmt: v => v.toFixed(3) },
    { label: 'High Risk', v1: p1Stats.highRisk, v2: p2Stats.highRisk, fmt: v => v.toString() },
    { label: 'Med Risk', v1: p1Stats.medRisk, v2: p2Stats.medRisk, fmt: v => v.toString() },
  ];

  return (
    <div className="glass-card" id="comparative-view">
      <div className="flex items-center gap-2 mb-4">
        <GitCompare size={16} className="text-teal-400" />
        <h3 className="text-sm font-semibold t-heading">Comparative Analytics</h3>
        <span className="ml-auto text-[10px] t-muted">First half vs Second half</span>
      </div>

      {/* Stats comparison table */}
      <div className="rounded-xl overflow-hidden mb-4" style={{ border: '1px solid var(--card-border)' }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ background: 'var(--panel-bg)' }}>
              <th className="px-3 py-2 text-left t-muted font-medium">Metric</th>
              <th className="px-3 py-2 text-center text-blue-400 font-medium">Period 1 ({p1Stats.frames}f)</th>
              <th className="px-3 py-2 text-center text-purple-400 font-medium">Period 2 ({p2Stats.frames}f)</th>
              <th className="px-3 py-2 text-right t-muted font-medium">Change</th>
            </tr>
          </thead>
          <tbody>
            {metrics.map((m, i) => (
              <tr key={i} style={{ borderBottom: '1px solid var(--panel-border)' }}>
                <td className="px-3 py-2 t-muted">{m.label}</td>
                <td className="px-3 py-2 text-center t-heading font-semibold">{m.fmt(m.v1)}</td>
                <td className="px-3 py-2 text-center t-heading font-semibold">{m.fmt(m.v2)}</td>
                <td className={`px-3 py-2 text-right font-bold ${deltaColor(m.v1, m.v2)}`}>
                  {delta(m.v1, m.v2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Overlay chart */}
      <div className="h-[180px]">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" />
            <XAxis dataKey="index" tick={{ fill: 'var(--color-text-muted)', fontSize: 9 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: 'var(--color-text-muted)', fontSize: 10 }} axisLine={false} tickLine={false} />
            <Tooltip
              contentStyle={{
                background: 'var(--tooltip-bg)',
                border: '1px solid var(--tooltip-border)',
                borderRadius: '10px',
                fontSize: '11px',
              }}
            />
            <Legend />
            <Line type="monotone" dataKey="Period 1" stroke="#6366f1" strokeWidth={2} dot={false} connectNulls={false} />
            <Line type="monotone" dataKey="Period 2" stroke="#a855f7" strokeWidth={2} dot={false} connectNulls={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
