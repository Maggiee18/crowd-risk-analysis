import React from 'react';
import {
  Activity, Users, Eye, AlertTriangle, BarChart3,
  Map, Brain, Shield, Play, Settings, LogOut, Download, Flame
} from 'lucide-react';

const navItems = [
  { id: 'dashboard', label: 'Dashboard', icon: Activity },
  { id: 'video', label: 'Live Feed', icon: Eye },
  { id: 'analytics', label: 'Analytics', icon: BarChart3 },
  { id: 'zones', label: 'Zone Map', icon: Map },
  { id: 'heatmap', label: 'Heatmap', icon: Flame },
  { id: 'alerts', label: 'Alerts', icon: AlertTriangle },
  { id: 'predictions', label: 'Predictions', icon: Brain },
  { id: 'control', label: 'Control', icon: Shield },
  { id: 'simulation', label: 'Simulation', icon: Play },
  { id: 'settings', label: 'Settings', icon: Settings },
];

export default function Sidebar({ activePage, onPageChange, isConnected, peopleCount }) {
  return (
    <aside
      className="fixed left-0 top-0 h-screen w-64 backdrop-blur-xl flex flex-col z-50 transition-colors duration-300"
      style={{
        backgroundColor: 'var(--card-bg)',
        borderRight: '1px solid var(--card-border)',
      }}
    >
      {/* Logo */}
      <div className="p-6" style={{ borderBottom: '1px solid var(--card-border)' }}>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-brand-600 flex items-center justify-center shadow-lg shadow-brand-600/30">
            <Users size={20} className="text-white" />
          </div>
          <div>
            <h1 className="text-sm font-bold t-heading tracking-wide">CrowdGuard</h1>
            <p className="text-[10px] t-muted uppercase tracking-widest">AI Risk System</p>
          </div>
        </div>
      </div>

      {/* Status */}
      <div
        className="px-4 py-3 mx-3 mt-3 rounded-xl transition-colors duration-300"
        style={{
          background: 'var(--panel-bg)',
          border: '1px solid var(--panel-border)',
        }}
      >
        <div className="flex items-center gap-2 mb-1">
          <span className={`pulse-dot ${isConnected ? 'online' : 'offline'}`} />
          <span className="text-xs t-muted">
            {isConnected ? 'Connected' : 'Disconnected'}
          </span>
        </div>
        <div className="text-lg font-bold t-heading">
          {peopleCount || 0}{' '}
          <span className="text-xs t-muted font-normal">people</span>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
        {navItems.map(item => {
          const Icon = item.icon;
          const isActive = activePage === item.id;
          return (
            <button
              key={item.id}
              onClick={() => onPageChange(item.id)}
              className={`sidebar-link w-full ${isActive ? 'active' : ''}`}
            >
              <Icon size={18} className={isActive ? 'text-brand-400' : ''} />
              <span>{item.label}</span>
              {item.id === 'alerts' && (
                <span className="ml-auto w-5 h-5 rounded-full bg-red-500/20 text-red-400 text-[10px] flex items-center justify-center font-bold">!</span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 space-y-2" style={{ borderTop: '1px solid var(--card-border)' }}>
        <button className="btn-ghost w-full justify-start text-xs">
          <Download size={14} /> Export Logs
        </button>
        <p className="text-[10px] t-muted text-center">v2.0 • AI-Powered</p>
      </div>
    </aside>
  );
}
