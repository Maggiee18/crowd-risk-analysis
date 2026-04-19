// Centralized API service
import axios from 'axios';

const API_BASE = (import.meta.env && import.meta.env.VITE_API_URL) || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── System ──────────────────────────────────────────────────────
export const getStatus = () => api.get('/status');
export const getCurrentState = () => api.get('/current_state');

// ── Alerts ──────────────────────────────────────────────────────
export const getAlerts = () => api.get('/alerts');

// ── History ─────────────────────────────────────────────────────
export const getHistory = (limit = 100) => api.get(`/history?limit=${limit}`);

// ── Predictions ─────────────────────────────────────────────────
export const getPredictions = () => api.get('/predictions');
export const getControlSuggestions = () => api.get('/control_suggestions');

// ── Zones ───────────────────────────────────────────────────────
export const getZones = () => api.get('/zones');

// ── Simulation ──────────────────────────────────────────────────
export const getScenarios = () => api.get('/scenarios');
export const startSimulation = (scenario) => api.post('/simulate', { scenario });
export const stopSimulation = () => api.post('/simulate/stop');

// ── Frame Processing ────────────────────────────────────────────
export const processFrame = (imageBase64) => api.post('/process_frame', { image: imageBase64 });

// ── Auth ────────────────────────────────────────────────────────
export const login = (username, password) => api.post('/auth/login', { username, password });

// ── Training ────────────────────────────────────────────────────
export const trainLSTM = (epochs = 30, n_scenarios = 30) =>
  api.post('/train/lstm', { epochs, n_scenarios });
export const trainRL = () => api.post('/train/rl');

// ── Privacy ─────────────────────────────────────────────────────
export const togglePrivacy = (enabled) =>
  api.get(`/privacy/toggle${enabled !== undefined ? `?enabled=${enabled}` : ''}`);

// ── Logs ────────────────────────────────────────────────────────
export const downloadLogs = () => api.get('/logs/download', { responseType: 'blob' });

export default api;
