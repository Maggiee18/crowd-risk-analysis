import React from 'react';
import {
  LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Legend
} from 'recharts';
import { BarChart3 } from 'lucide-react';

export default function CrowdChart({ data = [], title = 'Crowd Analytics' }) {
  // Format data for display
  const chartData = data.map((d, i) => ({
    time: d.timestamp
      ? new Date(d.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : `T${i}`,
    'People Count': d.people_count || d.peopleCount || 0,
    'Density': Math.round((d.density || 0) * 100) / 100,
    'Risk Score': d.risk_level === 'high' ? 3 : d.risk_level === 'medium' ? 2 : 1,
    'Speed': Math.round((d.avg_speed || 0) * 10) / 10,
  }));

  return (
    <div className="glass-card">
      <div className="flex items-center gap-2 mb-4">
        <BarChart3 size={18} className="text-brand-400" />
        <h3 className="text-sm font-semibold t-heading">{title}</h3>
        <span className="ml-auto text-[10px] t-muted">{chartData.length} pts</span>
      </div>

      {chartData.length === 0 ? (
        <div className="h-[260px] flex items-center justify-center t-muted text-sm">
          Collecting data…
        </div>
      ) : (
        <div className="chart-container">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 5, right: 5, left: -20, bottom: 5 }}>
              <defs>
                <linearGradient id="colorCount" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#6366f1" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#6366f1" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorRisk" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#ef4444" stopOpacity={0.3} />
                  <stop offset="100%" stopColor="#ef4444" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis
                dataKey="time"
                tick={{ fill: '#64748b', fontSize: 10 }}
                axisLine={{ stroke: 'rgba(255,255,255,0.06)' }}
                tickLine={false}
                interval="preserveStartEnd"
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
                  borderRadius: '12px',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                  fontSize: '12px',
                }}
                labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
              />
              <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
              <Area
                type="monotone"
                dataKey="People Count"
                stroke="#6366f1"
                fill="url(#colorCount)"
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4, fill: '#6366f1' }}
              />
              <Line
                type="monotone"
                dataKey="Risk Score"
                stroke="#ef4444"
                strokeWidth={2}
                dot={false}
                strokeDasharray="5 3"
              />
              <Line
                type="monotone"
                dataKey="Speed"
                stroke="#10b981"
                strokeWidth={1.5}
                dot={false}
                opacity={0.6}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
