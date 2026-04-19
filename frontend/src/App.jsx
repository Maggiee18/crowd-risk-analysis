import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Users, Activity, Gauge, Clock, Wifi, WifiOff,
  TrendingUp, Eye, BarChart3, Zap, Flame, AlertTriangle as AlertIcon,
  Sun, Moon, ArrowUpRight
} from 'lucide-react';
import './index.css';

// Components
import Sidebar from './components/Sidebar';
import StatsCard from './components/StatsCard';
import VideoFeed from './components/VideoFeed';
import RiskGauge from './components/RiskGauge';
import AlertPanel from './components/AlertPanel';
import CrowdChart from './components/CrowdChart';
import PredictionChart from './components/PredictionChart';
import ZoneMap from './components/ZoneMap';
import ControlPanel from './components/ControlPanel';
import SimulationPanel from './components/SimulationPanel';
import HeatmapPanel from './components/HeatmapPanel';
import LoginPage from './components/LoginPage';

// New Feature Components
import IncidentReport from './components/IncidentReport';
import AudioAlerts from './components/AudioAlerts';
import FlowOverlay from './components/FlowOverlay';
import SessionTimeline from './components/SessionTimeline';
import RiskExplainer from './components/RiskExplainer';
import ZoneEditor from './components/ZoneEditor';
import ComparativeView from './components/ComparativeView';
import NLQueryPanel from './components/NLQueryPanel';
import EmergencyPanel from './components/EmergencyPanel';
import SentimentGauge from './components/SentimentGauge';
import WhatIfSimulator from './components/WhatIfSimulator';

// Hooks and services
import { useWebSocket } from './hooks/useWebSocket';
import * as api from './services/api';

function App() {
  // ── Auth ─────────────────────────────────────────────────────
  const [isLoggedIn, setIsLoggedIn] = useState(false);

  // ── UI State ─────────────────────────────────────────────────
  const [activePage, setActivePage] = useState('dashboard');
  const [currentTime, setCurrentTime] = useState(new Date().toLocaleTimeString());

  // Update clock every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date().toLocaleTimeString()), 1000);
    return () => clearInterval(timer);
  }, []);
  const [theme, setTheme] = useState(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('crowdguard-theme') || 'dark';
    }
    return 'dark';
  });

  // ── Apply theme ──────────────────────────────────────────────
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('crowdguard-theme', theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme(prev => prev === 'dark' ? 'light' : 'dark');
  }, []);

  // ── Data State — ALL driven by real-time frame processing ────
  const [systemStatus, setSystemStatus] = useState(null);
  const [currentState, setCurrentState] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [alertHistory, setAlertHistory] = useState([]);
  const [history, setHistory] = useState([]);
  const [zones, setZones] = useState([]);
  const [predictions, setPredictions] = useState({});
  const [suggestions, setSuggestions] = useState([]);
  const [privacyEnabled, setPrivacyEnabled] = useState(true);
  const [heatmapData, setHeatmapData] = useState(null);

  // ── Input source tracking ───────────────────────────────────
  const [inputSource, setInputSource] = useState('none'); // 'none' | 'camera' | 'upload' | 'simulation'

  // ── Simulation ───────────────────────────────────────────────
  const [simRunning, setSimRunning] = useState(false);
  const [simProgress, setSimProgress] = useState(0);

  // ── Processing guard ─────────────────────────────────────────
  const [isProcessingFrame, setIsProcessingFrame] = useState(false);
  const processingRef = useRef(false);

  // ═══════════════════════════════════════════════════════════════
  // UNIFIED STATE UPDATER — every data source funnels through here
  // ═══════════════════════════════════════════════════════════════
  const applyFrameResult = useCallback((d, frameB64) => {
    // Update current state (feeds video display + all panels)
    setCurrentState(prev => ({
      ...prev,
      current_frame: d.frame_b64 || frameB64 || prev?.current_frame,
      people_count: d.people_count ?? prev?.people_count ?? 0,
      detections: d.detections || prev?.detections || [],
      features: d.features || prev?.features || {},
      risk_prediction: d.risk_prediction || prev?.risk_prediction || {},
      anomaly_detection: d.anomaly_detection || prev?.anomaly_detection || {},
      predictions: d.predictions || prev?.predictions,
      control_suggestions: d.control_suggestions || prev?.control_suggestions,
      tracks: d.tracks || prev?.tracks,
      tracking_summary: d.tracking_summary || prev?.tracking_summary,
    }));

    // Heatmap
    if (d.heatmap) setHeatmapData(d.heatmap);

    // Predictions
    if (d.predictions) setPredictions(d.predictions);

    // Control suggestions
    if (d.control_suggestions) setSuggestions(d.control_suggestions);

    // Alerts
    if (d.alerts?.length > 0) {
      setAlerts(d.alerts);
      setAlertHistory(prev => [...prev.slice(-50), ...d.alerts]);
    }

    // History (time-series for charts)
    setHistory(prev => {
      const entry = {
        timestamp: d.timestamp || new Date().toISOString(),
        people_count: d.people_count || 0,
        density: d.features?.density_per_frame || 0,
        risk_level: d.risk_prediction?.risk_level || 'low',
        avg_speed: d.tracking_summary?.average_speed || d.features?.avg_magnitude || 0,
      };
      return [...prev.slice(-200), entry];
    });
  }, []);

  // ═══════════════════════════════════════════════════════════════
  // FRAME HANDLER — called by VideoFeed for camera & uploaded video
  // ═══════════════════════════════════════════════════════════════
  const handleFrameReady = useCallback(async (base64Frame, source) => {
    // Guard: skip if still processing previous frame
    if (processingRef.current) return;
    processingRef.current = true;
    setIsProcessingFrame(true);

    try {
      const res = await api.processFrame(base64Frame);
      const d = res.data;
      // The backend returns the full pipeline result (detection, risk, anomaly, LSTM, RL, heatmap, alerts)
      applyFrameResult(d, base64Frame);
    } catch (e) {
      console.error(`Frame processing error (${source}):`, e);
    } finally {
      processingRef.current = false;
      setIsProcessingFrame(false);
    }
  }, [applyFrameResult]);

  // ═══════════════════════════════════════════════════════════════
  // WEBSOCKET — for simulation real-time stream
  // ═══════════════════════════════════════════════════════════════
  const handleWsMessage = useCallback((msg) => {
    if (msg.type === 'simulation_frame' && msg.data) {
      const d = msg.data;

      // When simulation starts, mark source
      if (inputSource !== 'simulation') setInputSource('simulation');

      applyFrameResult(d, d.frame_b64);

      // Simulation progress
      if (d.simulation) {
        setSimProgress(d.simulation.progress || 0);
        if (!d.simulation.is_running) {
          setSimRunning(false);
          setInputSource('none');
        }
      }
    } else if (msg.type === 'simulation_complete') {
      setSimRunning(false);
      setInputSource('none');
    }
  }, [applyFrameResult, inputSource]);

  const { isConnected } = useWebSocket(handleWsMessage);

  // ═══════════════════════════════════════════════════════════════
  // POLLING — fetches system state & zones periodically
  // ═══════════════════════════════════════════════════════════════
  useEffect(() => {
    if (!isLoggedIn) return;

    const fetchStatus = async () => {
      try {
        const [statusRes, zoneRes] = await Promise.allSettled([
          api.getStatus(),
          api.getZones(),
        ]);
        if (statusRes.status === 'fulfilled') setSystemStatus(statusRes.value.data);
        if (zoneRes.status === 'fulfilled') setZones(zoneRes.value.data.zones || []);
      } catch (e) {
        console.error('Polling error:', e);
      }
    };

    // Also do a full state fetch on login (to pick up any existing state)
    const fetchInitialState = async () => {
      try {
        const [stateRes, alertRes, histRes] = await Promise.allSettled([
          api.getCurrentState(),
          api.getAlerts(),
          api.getHistory(200),
        ]);
        if (stateRes.status === 'fulfilled') {
          const s = stateRes.value.data;
          setCurrentState(s);
          if (s.predictions) setPredictions(s.predictions);
          if (s.control_suggestions) setSuggestions(s.control_suggestions);
          if (s.heatmap) setHeatmapData(s.heatmap);
        }
        if (alertRes.status === 'fulfilled') {
          setAlerts(alertRes.value.data.active_alerts || []);
          setAlertHistory(alertRes.value.data.recent_history || []);
        }
        if (histRes.status === 'fulfilled') setHistory(histRes.value.data.data || []);
      } catch (e) {
        console.error('Initial fetch error:', e);
      }
    };

    fetchStatus();
    fetchInitialState();

    // Poll status + zones at a slower rate (these don't need per-frame update)
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [isLoggedIn]);

  // ═══════════════════════════════════════════════════════════════
  // HANDLERS
  // ═══════════════════════════════════════════════════════════════
  const handleLogin = async (username, password) => {
    const res = await api.login(username, password);
    if (res.data.status === 'success') {
      setIsLoggedIn(true);
    } else {
      throw new Error('Login failed');
    }
  };

  const handleStartSim = async (scenario) => {
    try {
      await api.startSimulation(scenario);
      setSimRunning(true);
      setSimProgress(0);
      setInputSource('simulation');
    } catch (e) {
      console.error('Failed to start simulation:', e);
    }
  };

  const handleStopSim = async () => {
    try {
      await api.stopSimulation();
      setSimRunning(false);
      setInputSource('none');
    } catch (e) {
      console.error('Failed to stop simulation:', e);
    }
  };

  const handleTogglePrivacy = async () => {
    try {
      const res = await api.togglePrivacy(!privacyEnabled);
      setPrivacyEnabled(res.data.enabled);
    } catch (e) {
      setPrivacyEnabled(!privacyEnabled);
    }
  };

  const handleSourceChange = useCallback((source) => {
    setInputSource(source);
  }, []);

  // ── Derived values ───────────────────────────────────────────
  const peopleCount = currentState?.people_count || 0;
  const riskLevel = currentState?.risk_prediction?.risk_level || 'low';
  const riskConf = currentState?.risk_prediction?.confidence || 0;
  const frameData = currentState?.current_frame || null;
  const features = currentState?.features || {};
  const anomaly = currentState?.anomaly_detection || {};
  const fps = systemStatus?.performance?.current_fps || 0;

  // ── Login Screen ─────────────────────────────────────────────
  if (!isLoggedIn) {
    return <LoginPage onLogin={handleLogin} />;
  }

  // ── Shared VideoFeed props ───────────────────────────────────
  const videoFeedProps = {
    frameData,
    isConnected,
    privacyEnabled,
    onTogglePrivacy: handleTogglePrivacy,
    onFrameReady: handleFrameReady,
    isProcessing: isProcessingFrame,
    inputSource,
    onSourceChange: handleSourceChange,
  };

  // ── Main Dashboard ───────────────────────────────────────────
  return (
    <div className="min-h-screen bg-grid" style={{ backgroundColor: 'var(--color-bg)' }}>
      {/* Sidebar */}
      <Sidebar
        activePage={activePage}
        onPageChange={setActivePage}
        isConnected={isConnected}
        peopleCount={peopleCount}
      />

      {/* Main content */}
      <main className="ml-64 p-6 min-h-screen">
        {/* Top bar */}
        <header className="flex items-center justify-between mb-6">
          <div>
            <h2 className="text-xl font-bold t-heading capitalize">{activePage.replace('_', ' ')}</h2>
            <p className="text-xs t-muted mt-0.5">
              Real-time AI-powered crowd monitoring • {inputSource !== 'none' ? `Source: ${inputSource}` : 'No input active'}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {/* New Features: Emergency Panel and Incident Report */}
            <EmergencyPanel 
              currentState={currentState} 
              history={history} 
              alerts={alerts} 
            />
            <IncidentReport 
              currentState={currentState} 
              history={history} 
              alerts={alerts} 
              heatmapData={heatmapData} 
              theme={theme} 
            />

            {/* Anomaly indicator */}
            {anomaly.is_anomaly && (
              <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-red-500/15 border border-red-500/20 text-xs text-red-400 font-semibold animate-pulse">
                <AlertIcon size={12} /> ANOMALY
              </div>
            )}

            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-300"
              style={{
                background: 'var(--panel-bg)',
                border: '1px solid var(--card-border)',
                color: 'var(--color-text-muted)',
              }}
              title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
              id="theme-toggle-btn"
            >
              {theme === 'dark' ? (
                <><Sun size={14} className="text-amber-400" /><span>Light</span></>
              ) : (
                <><Moon size={14} className="text-indigo-500" /><span>Dark</span></>
              )}
            </button>

            {/* Connection status */}
            <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--color-text-muted)' }}>
              {isConnected ? (
                <><Wifi size={14} className="text-emerald-400" /><span className="text-emerald-400">Live</span></>
              ) : (
                <><WifiOff size={14} className="text-red-400" /><span className="text-red-400">Offline</span></>
              )}
            </div>

            {/* Clock */}
            <div
              className="flex items-center gap-2 px-3 py-1.5 rounded-xl text-xs"
              style={{
                background: 'var(--panel-bg)',
                border: '1px solid var(--card-border)',
                color: 'var(--color-text-muted)',
              }}
            >
              <Clock size={12} />
              {currentTime}
            </div>
          </div>
        </header>

        {/* ============ DASHBOARD ============ */}
        {activePage === 'dashboard' && (
          <div className="space-y-6 animate-fade-in">
            {/* Stats row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatsCard
                title="People Count"
                value={peopleCount}
                icon={Users}
                color={peopleCount > 25 ? 'red' : peopleCount > 12 ? 'amber' : 'green'}
                trend={features.growth_rate}
                subtitle={`${Object.keys(currentState?.tracks || {}).length} unique tracks`}
              />
              <StatsCard
                title="Density"
                value={(features.density_per_frame || 0).toFixed(2)}
                icon={Activity}
                color="brand"
                subtitle="People / m²"
              />
              <StatsCard
                title="Avg Speed"
                value={(features.avg_magnitude || 0).toFixed(1)}
                icon={Zap}
                color="purple"
                subtitle="px/sec"
              />
              <StatsCard
                title="Processing"
                value={`${fps.toFixed(0)} FPS`}
                icon={Gauge}
                color="green"
                subtitle={`Frame #${systemStatus?.frame_count || 0}`}
              />
            </div>

            {/* Video + Risk + Alerts */}
            {/* Video + Risk + Alerts */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2 relative group">
                <VideoFeed {...videoFeedProps} />
                <button 
                  onClick={() => setActivePage('video')} 
                  className="absolute top-4 right-4 p-2 rounded-xl bg-black/40 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110 shadow-lg"
                  title="Go to Live Feed"
                >
                  <ArrowUpRight size={16} />
                </button>
              </div>
              <div className="space-y-4">
                <RiskGauge riskLevel={riskLevel} confidence={riskConf} />
                <div className="relative group">
                  <AlertPanel alerts={alerts} alertHistory={alertHistory} />
                  <button 
                    onClick={() => setActivePage('alerts')} 
                    className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                    title="Go to Alerts"
                  >
                    <ArrowUpRight size={14} />
                  </button>
                </div>
                <RiskExplainer 
                  features={features} 
                  riskPrediction={currentState?.risk_prediction} 
                  anomaly={anomaly} 
                  peopleCount={peopleCount} 
                  history={history} 
                />
                <SentimentGauge 
                  features={features} 
                  anomaly={anomaly} 
                  peopleCount={peopleCount} 
                />
              </div>
            </div>

            {/* Advanced Analytics Row */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-1">
                <FlowOverlay features={features} detections={currentState?.detections || []} />
              </div>
              <div className="lg:col-span-2">
                <SessionTimeline history={history} alerts={alerts} />
              </div>
            </div>

            {/* Heatmap + Crowd chart */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="relative group">
                <HeatmapPanel
                  heatmapData={heatmapData}
                  detections={currentState?.detections || []}
                  frameWidth={640}
                  frameHeight={480}
                />
                <button 
                  onClick={() => setActivePage('heatmap')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Heatmap"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
              <div className="relative group">
                <CrowdChart data={history} title="Crowd Analytics" />
                <button 
                  onClick={() => setActivePage('analytics')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Analytics"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
            </div>

            {/* Predictions + Zones */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="relative group">
                <PredictionChart predictions={predictions} history={history} />
                <button 
                  onClick={() => setActivePage('predictions')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Predictions"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
              <div className="relative group">
                <ZoneMap zones={zones} />
                <button 
                  onClick={() => setActivePage('zones')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Zones"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
            </div>

            {/* Control + Simulation */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <div className="relative group">
                <ControlPanel suggestions={suggestions} />
                <button 
                  onClick={() => setActivePage('control')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Control"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
              <div className="relative group">
                <SimulationPanel
                  onStart={handleStartSim}
                  onStop={handleStopSim}
                  isRunning={simRunning}
                  progress={simProgress}
                />
                <button 
                  onClick={() => setActivePage('simulation')} 
                  className="absolute top-4 right-4 p-1.5 rounded-lg bg-black/20 hover:bg-brand-500 text-white backdrop-blur-md opacity-0 group-hover:opacity-100 transition-all z-10 hover:scale-110"
                  title="Go to Simulation"
                >
                  <ArrowUpRight size={14} />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ============ LIVE FEED ============ */}
        {activePage === 'video' && (
          <div className="animate-fade-in space-y-4">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
              <div className="lg:col-span-2">
                <VideoFeed {...videoFeedProps} />
              </div>
              <div className="space-y-4">
                <RiskGauge riskLevel={riskLevel} confidence={riskConf} />
                <AlertPanel alerts={alerts} alertHistory={alertHistory} />
              </div>
            </div>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <HeatmapPanel
                heatmapData={heatmapData}
                detections={currentState?.detections || []}
                frameWidth={640}
                frameHeight={480}
              />
              <CrowdChart data={history} title="Live Analytics" />
            </div>
          </div>
        )}

        {/* ============ ANALYTICS ============ */}
        {activePage === 'analytics' && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatsCard title="Total Frames" value={systemStatus?.frame_count || 0} icon={BarChart3} color="brand" />
              <StatsCard title="People Count" value={peopleCount} icon={Users} color="green" />
              <StatsCard title="Peak Count" value={Math.max(...history.map(h => h.people_count || 0), 0)} icon={TrendingUp} color="amber" />
              <StatsCard title="Alerts Fired" value={alertHistory.length} icon={Eye} color="red" />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <CrowdChart data={history} title="Historical Crowd Data" />
              <ComparativeView history={history} />
            </div>
            <PredictionChart predictions={predictions} history={history} />
          </div>
        )}

        {/* ============ ZONES ============ */}
        {activePage === 'zones' && (
          <div className="max-w-3xl animate-fade-in space-y-4">
            <ZoneEditor 
              frameData={frameData} 
              frameWidth={640} 
              frameHeight={480} 
              onZonesUpdate={() => {}} 
            />
            <ZoneMap zones={zones} />
          </div>
        )}

        {/* ============ HEATMAP ============ */}
        {activePage === 'heatmap' && (
          <div className="max-w-3xl animate-fade-in space-y-4">
            <HeatmapPanel
              heatmapData={heatmapData}
              detections={currentState?.detections || []}
              frameWidth={640}
              frameHeight={480}
            />
            <ZoneMap zones={zones} />
          </div>
        )}

        {/* ============ ALERTS ============ */}
        {activePage === 'alerts' && (
          <div className="max-w-2xl animate-fade-in">
            <AlertPanel alerts={alerts} alertHistory={alertHistory} />
          </div>
        )}

        {/* ============ PREDICTIONS ============ */}
        {activePage === 'predictions' && (
          <div className="max-w-3xl animate-fade-in">
            <PredictionChart predictions={predictions} history={history} />
          </div>
        )}

        {/* ============ CONTROL ============ */}
        {activePage === 'control' && (
          <div className="max-w-2xl animate-fade-in">
            <ControlPanel suggestions={suggestions} />
          </div>
        )}

        {/* ============ SIMULATION ============ */}
        {activePage === 'simulation' && (
          <div className="space-y-4 animate-fade-in">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <SimulationPanel
                onStart={handleStartSim}
                onStop={handleStopSim}
                isRunning={simRunning}
                progress={simProgress}
              />
              <VideoFeed {...videoFeedProps} />
            </div>
            
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              <CrowdChart data={history} title="Simulation Analytics" />
              <WhatIfSimulator currentState={currentState} features={features} />
            </div>
          </div>
        )}

        {/* ============ SETTINGS ============ */}
        {activePage === 'settings' && (
          <div className="max-w-xl animate-fade-in">
            <div className="glass-card space-y-4">
              <h3 className="text-sm font-semibold t-heading">System Settings</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
                  <span className="text-sm t-body">Privacy Filter</span>
                  <button onClick={handleTogglePrivacy} className={`px-3 py-1 rounded-lg text-xs font-semibold ${privacyEnabled ? 'bg-emerald-500/20 text-emerald-400' : 'bg-red-500/20 text-red-400'}`}>
                    {privacyEnabled ? 'Enabled' : 'Disabled'}
                  </button>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
                  <span className="text-sm t-body">WebSocket</span>
                  <span className={`text-xs font-semibold ${isConnected ? 'text-emerald-400' : 'text-red-400'}`}>
                    {isConnected ? 'Connected' : 'Disconnected'}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
                  <span className="text-sm t-body">Backend</span>
                  <span className={`text-xs font-semibold ${systemStatus?.initialized ? 'text-emerald-400' : 'text-red-400'}`}>
                    {systemStatus?.initialized ? 'Online' : 'Offline'}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
                  <span className="text-sm t-body">Input Source</span>
                  <span className="text-xs font-semibold text-brand-400 capitalize">
                    {inputSource}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-xl" style={{ background: 'var(--panel-bg)', border: '1px solid var(--panel-border)' }}>
                  <span className="text-sm t-body">Theme</span>
                  <span className="text-xs font-semibold text-brand-400 capitalize">
                    {theme}
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Global Hidden / Overlay Components */}
      <AudioAlerts 
        riskLevel={riskLevel} 
        anomalyDetected={anomaly?.is_anomaly} 
        enabled={true} 
      />
      
      <NLQueryPanel 
        history={history} 
        alerts={alerts} 
        currentState={currentState} 
        predictions={predictions} 
      />
    </div>
  );
}

export default App;
