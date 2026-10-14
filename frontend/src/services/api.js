// Centralized API service
import axios from 'axios';

const API_BASE = (import.meta.env && import.meta.env.VITE_API_URL) || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_BASE}/api`,
  timeout: 15000,
  headers: { 'Content-Type': 'application/json' },
});

// ── Session (login token) ───────────────────────────────────────
// The token expires on the server; we keep it so a page refresh stays
// logged in, and drop it as soon as the server says it is no longer valid.
const SESSION_KEY = 'crowdguard-session';

function loadSession() {
  try {
    const s = JSON.parse(localStorage.getItem(SESSION_KEY));
    if (s && s.token && s.expires_at * 1000 > Date.now()) return s;
  } catch (e) { /* storage unavailable */ }
  return null;
}

let session = loadSession();
let onUnauthorized = null;

export const getSession = () => session;
export const getToken = () => (session ? session.token : null);
export const setSession = (s) => {
  session = s;
  try {
    if (s) localStorage.setItem(SESSION_KEY, JSON.stringify(s));
    else localStorage.removeItem(SESSION_KEY);
  } catch (e) { /* storage unavailable */ }
};
export const setUnauthorizedHandler = (fn) => { onUnauthorized = fn; };

// ── Camera ──────────────────────────────────────────────────────
// Every request is about the camera picked in the dashboard (?camera=...),
// unless it names one itself. Endpoints without cameras ignore it.
let activeCamera = 'main';
export const getCamera = () => activeCamera;
export const setCamera = (id) => { activeCamera = id || 'main'; };

api.interceptors.request.use((config) => {
  if (session && session.token) config.headers.Authorization = `Bearer ${session.token}`;
  config.params = { camera: activeCamera, ...(config.params || {}) };
  return config;
});

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const status = err.response && err.response.status;
    const isLogin = err.config && String(err.config.url).includes('/auth/login');
    if (status === 401 && !isLogin) {
      setSession(null);
      if (onUnauthorized) onUnauthorized();
    }
    return Promise.reject(err);
  }
);

export const errorMessage = (err, fallback = 'Something went wrong') =>
  (err && err.response && err.response.data && err.response.data.detail) || fallback;

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
export const getZoneDefinitions = () => api.get('/zones/definitions');
export const saveZones = (zones) => api.put('/zones', { zones });

// ── Simulation ──────────────────────────────────────────────────
export const getScenarios = () => api.get('/scenarios');
export const startSimulation = (scenario) => api.post('/simulate', { scenario });
export const stopSimulation = () => api.post('/simulate/stop');

// ── Frame Processing ────────────────────────────────────────────
// Frames from this browser always belong to the browser camera ('main')
export const processFrame = (imageBase64, returnFrame = true) =>
  api.post('/process_frame', { image: imageBase64, return_frame: returnFrame }, { params: { camera: 'main' } });

// ── Cameras ─────────────────────────────────────────────────────
export const listCameras = (thumbs = true) => api.get('/cameras', { params: { thumbs } });
export const addCamera = (name, source) => api.post('/cameras', { name, source });
export const updateCamera = (id, changes) => api.put(`/cameras/${encodeURIComponent(id)}`, changes);
export const deleteCamera = (id) => api.delete(`/cameras/${encodeURIComponent(id)}`);

// ── Auth ────────────────────────────────────────────────────────
export const login = (username, password) => api.post('/auth/login', { username, password });
export const getMe = () => api.get('/auth/me');
export const changePassword = (oldPassword, newPassword) =>
  api.post('/auth/password', { old_password: oldPassword, new_password: newPassword });
export const listUsers = () => api.get('/auth/users');
export const createUser = (username, password, role) => api.post('/auth/users', { username, password, role });
export const deleteUser = (username) => api.delete(`/auth/users/${encodeURIComponent(username)}`);

// ── Notifications (admin) ───────────────────────────────────────
export const getNotifySettings = () => api.get('/notify');
export const updateNotifySettings = (settings) => api.put('/notify', { settings });
export const sendTestNotification = () => api.post('/notify/test', {}, { timeout: 45000 });
export const findTelegramChat = () => api.post('/notify/telegram/find_chat', {}, { timeout: 30000 });

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
