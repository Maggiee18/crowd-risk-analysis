import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

export default function StatsCard({ title, value, subtitle, icon: Icon, trend, color = 'brand' }) {
  const colorMap = {
    brand: 'from-brand-600/20 to-brand-800/10 border-brand-500/15',
    green: 'from-emerald-600/20 to-emerald-800/10 border-emerald-500/15',
    amber: 'from-amber-600/20 to-amber-800/10 border-amber-500/15',
    red: 'from-red-600/20 to-red-800/10 border-red-500/15',
    purple: 'from-purple-600/20 to-purple-800/10 border-purple-500/15',
  };

  const iconBg = {
    brand: 'bg-brand-500/20 text-brand-400',
    green: 'bg-emerald-500/20 text-emerald-400',
    amber: 'bg-amber-500/20 text-amber-400',
    red: 'bg-red-500/20 text-red-400',
    purple: 'bg-purple-500/20 text-purple-400',
  };

  const TrendIcon = trend > 0 ? TrendingUp : trend < 0 ? TrendingDown : Minus;
  const trendColor = trend > 0 ? 'text-red-400' : trend < 0 ? 'text-emerald-400' : 'text-slate-500';

  return (
    <div className={`glass-card bg-gradient-to-br ${colorMap[color]} group`}>
      <div className="flex items-start justify-between mb-3">
        <div className={`w-10 h-10 rounded-xl ${iconBg[color]} flex items-center justify-center transition-transform group-hover:scale-110`}>
          {Icon && <Icon size={20} />}
        </div>
        {trend !== undefined && (
          <div className={`flex items-center gap-1 text-xs ${trendColor}`}>
            <TrendIcon size={14} />
            <span>{Math.abs(trend).toFixed(1)}%</span>
          </div>
        )}
      </div>
      <div className="stat-number mb-1">{value}</div>
      <div className="text-xs t-muted font-medium uppercase tracking-wider">{title}</div>
      {subtitle && <div className="text-[11px] t-muted mt-1">{subtitle}</div>}
    </div>
  );
}
