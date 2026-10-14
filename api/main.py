"""
FastAPI Backend for Crowd Risk Detection System
Provides REST API + WebSocket for real-time crowd analysis.
"""

import os
import sys
import asyncio
import base64
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi import Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Add src to path
SRC_DIR = os.path.join(os.path.dirname(__file__), '..', 'src')
sys.path.insert(0, SRC_DIR)

from detector import PeopleDetector
from analyzer import CrowdAnalyzer
from predictor import RiskPredictor
from anomaly_detector import AnomalyDetector
from optimizer import AlertSystem, HeatmapGenerator, PerformanceMonitor
from mall_models import load_mall_models
from tracker import PersonTracker
from panic_detector import PanicDetector
from evacuation_detector import EvacuationDetector
from auth import UserStore, AuthError, ROLES
from zones import ZoneManager, zone_alerts
from notifier import Notifier
from lstm_predictor import LSTMCrowdPredictor
from rl_controller import CrowdControlAgent
from privacy import PrivacyFilter
from simulator import CrowdSimulator
from utils import blur_faces, generate_zone_grid

# ─── Logging ──────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s %(message)s')
logger = logging.getLogger("crowd_api")

# ─── FastAPI app ──────────────────────────────────────────────────
app = FastAPI(
    title="Crowd Risk Detection API",
    description="AI-powered crowd risk prediction and control system",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global state ────────────────────────────────────────────────
class SystemState:
    def __init__(self):
        self.detector: Optional[PeopleDetector] = None
        self.analyzer: Optional[CrowdAnalyzer] = None
        self.risk_predictor: Optional[RiskPredictor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.tracker: Optional[PersonTracker] = None
        self.lstm_predictor: Optional[LSTMCrowdPredictor] = None
        self.rl_agent: Optional[CrowdControlAgent] = None
        self.privacy_filter: Optional[PrivacyFilter] = None
        self.simulator: Optional[CrowdSimulator] = None
        self.alert_system: Optional[AlertSystem] = None
        self.panic_detector: Optional[PanicDetector] = None
        self.evacuation_detector: Optional[EvacuationDetector] = None
        self.stream_time = 0.0
        self.current_panic: Dict = {}
        self.heatmap_gen: Optional[HeatmapGenerator] = None
        self.perf_monitor: Optional[PerformanceMonitor] = None

        self.initialized = False
        self.processing = False
        self.simulation_active = False

        # Current data
        self.current_frame: Optional[np.ndarray] = None
        self.current_detections: List[Dict] = []
        self.current_tracks: Dict = {}
        self.current_features: Dict = {}
        self.current_risk: Dict = {}
        self.current_anomaly: Dict = {}
        self.current_alerts: List[Dict] = []
        self.current_predictions: Dict = {}
        self.current_suggestions: List[Dict] = []
        self.current_heatmap: Optional[List] = None

        # History
        self.frame_count = 0
        self.history: deque = deque(maxlen=500)
        self.alert_log: deque = deque(maxlen=200)

        # WebSocket clients
        self.ws_clients: List[WebSocket] = []

        # Accounts and login tokens (see src/auth.py)
        data_dir = os.environ.get("CROWDGUARD_DATA_DIR",
                                  os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.users = UserStore(data_dir)
        # Zones drawn in the dashboard (see src/zones.py)
        self.zones = ZoneManager(os.path.join(data_dir, 'zones.json'))
        self.current_zone_stats: List[Dict] = []
        # Telegram / email notifications for critical alerts (see src/notifier.py)
        self.notifier = Notifier(os.path.join(data_dir, 'notify.json'))

state = SystemState()


# ─── Initialization ──────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Initialize all system components on startup."""
    try:
        logger.info("Initializing Crowd Risk Detection System …")

        generated = state.users.ensure_admin()
        if generated:
            logger.warning("=" * 60)
            logger.warning("First start: created user 'admin' with password: %s", generated)
            logger.warning("Log in and change it, or set CROWDGUARD_ADMIN_PASSWORD")
            logger.warning("before the first start. This is shown only once.")
            logger.warning("=" * 60)

        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)

        # Core detection
        model_path = os.path.join(os.path.dirname(__file__), '..', 'yolov8n.pt')
        state.detector = PeopleDetector(
            model_path=model_path if os.path.exists(model_path) else 'yolov8n.pt',
            confidence_threshold=0.15,
            imgsz=1280
        )

        state.analyzer = CrowdAnalyzer(frame_shape=(480, 640), history_length=60)

        # ML models
        state.risk_predictor = RiskPredictor(model_type='random_forest')
        state.anomaly_detector = AnomalyDetector(method='isolation_forest', contamination=0.1)

        # Multi-object tracker
        state.tracker = PersonTracker(use_deep_sort=True, max_disappeared=30)
        state.panic_detector = PanicDetector()
        state.evacuation_detector = EvacuationDetector()

        # LSTM predictor
        state.lstm_predictor = LSTMCrowdPredictor(window_size=30, prediction_steps=10)

        # RL agent
        state.rl_agent = CrowdControlAgent()
        state.rl_agent.train_simulated(n_episodes=200)

        # Privacy filter
        state.privacy_filter = PrivacyFilter(blur_strength=51, enabled=True)

        # Simulator
        state.simulator = CrowdSimulator(frame_width=640, frame_height=480, fps=15)

        # Alert system
        state.alert_system = AlertSystem(
            risk_threshold=0.7, anomaly_threshold=0.6,
            capacity_limit=30, alert_ratio=0.8, trend_min_slope=0.5
        )

        # Heatmap generator
        state.heatmap_gen = HeatmapGenerator(grid_size=20)

        # Performance monitor
        state.perf_monitor = PerformanceMonitor()

        # Load models trained by training/train_models.py if they exist
        _load_trained_models(model_dir)

        # Otherwise fall back to a quick LSTM trained on synthetic data
        if not state.lstm_predictor.is_trained:
            try:
                result = state.lstm_predictor.train(epochs=1, n_scenarios=2)
                logger.info(f"LSTM predictor trained on synthetic data: {result.get('status')}")
            except Exception as e:
                logger.warning(f"LSTM training skipped: {e}")

        state.initialized = True
        logger.info("✓ System initialization complete!")

    except Exception as e:
        logger.error(f"Initialization error: {e}", exc_info=True)
        state.initialized = False


def _load_trained_models(model_dir: str):
    """Load the Mall-trained models from models/ (each one is optional)."""
    try:
        models = load_mall_models(model_dir)
    except Exception as e:
        logger.warning(f"Could not load trained models: {e}")
        models = {}
    if 'count_corrector' in models:
        state.detector.count_corrector = models['count_corrector']
        logger.info("Loaded count corrector")
    if 'risk_predictor' in models:
        state.risk_predictor = models['risk_predictor']
        logger.info("Loaded Mall-trained risk predictor")
    if 'anomaly_detector' in models:
        state.anomaly_detector = models['anomaly_detector']
        logger.info("Loaded Mall-trained anomaly detector")
    lstm_dir = os.path.join(model_dir, 'lstm')
    if os.path.exists(os.path.join(lstm_dir, 'lstm_meta.json')):
        try:
            state.lstm_predictor.load(lstm_dir)
            logger.info("Loaded Mall-trained LSTM")
        except Exception as e:
            logger.warning(f"Could not load LSTM: {e}")


# ─── Pydantic models ─────────────────────────────────────────────
class FrameInput(BaseModel):
    image: str  # base64 encoded
    # The dashboard's smooth view draws the live video itself and only needs
    # the analysis, so it can skip the processed image (saves a JPEG encode
    # and ~60 KB per frame)
    return_frame: bool = True

class SimulationRequest(BaseModel):
    scenario: str = 'normal'

class LoginRequest(BaseModel):
    username: str
    password: str

class TrainRequest(BaseModel):
    epochs: int = 30
    n_scenarios: int = 30


# ─── Auth helpers ────────────────────────────────────────────────
def current_user(authorization: Optional[str] = Header(None)) -> Dict:
    """Any logged-in user. Expects 'Authorization: Bearer <token>'."""
    token = (authorization or "").removeprefix("Bearer ").strip()
    try:
        return state.users.verify_token(token)
    except AuthError as e:
        raise HTTPException(status_code=401, detail=str(e))


def admin_user(user: Dict = Depends(current_user)) -> Dict:
    """Logged-in admin only (controls the system, manages users)."""
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin only")
    return user


LOGGED_IN = [Depends(current_user)]
ADMIN_ONLY = [Depends(admin_user)]


# ─── REST API Endpoints ─────────────────────────────────────────
@app.get("/api/status")
async def get_status():
    """Get system status and component health."""
    return {
        "status": "ok" if state.initialized else "initializing",
        "initialized": state.initialized,
        "processing": state.processing,
        "simulation_active": state.simulation_active,
        "frame_count": state.frame_count,
        "ws_clients": len(state.ws_clients),
        "components": {
            "detector": state.detector is not None,
            "analyzer": state.analyzer is not None,
            "risk_predictor": state.risk_predictor is not None,
            "anomaly_detector": state.anomaly_detector is not None,
            "tracker": state.tracker is not None,
            "lstm_predictor": state.lstm_predictor is not None,
            "rl_agent": state.rl_agent is not None,
            "privacy_filter": state.privacy_filter is not None,
            "simulator": state.simulator is not None,
        },
        "performance": state.perf_monitor.get_stats() if state.perf_monitor else {},
        "timestamp": datetime.now().isoformat()
    }


@app.post("/api/process_frame", dependencies=LOGGED_IN)
async def process_frame(data: FrameInput):
    """Process a single base64-encoded frame and return full AI analysis."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")

    try:
        result = _process_frame_internal(data.image)

        # Encode the stored frame (with privacy filter) back to base64
        # so the frontend can display the processed view
        if data.return_frame and state.current_frame is not None:
            display_frame = state.current_frame.copy()

            # Draw detection bounding boxes on the frame
            for det in result.get('detections', [])[:50]:
                x1, y1, x2, y2 = det.get('bbox', [0, 0, 0, 0])
                conf = det.get('confidence', 0)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{conf:.0%}"
                cv2.putText(display_frame, label, (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Draw people count
            cv2.putText(display_frame,
                        f"People: {result.get('people_count', 0)}",
                        (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Draw risk level
            risk = result.get('risk_prediction', {})
            rl = risk.get('risk_level', 'low').upper()
            rc = risk.get('confidence', 0)
            color = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 'HIGH': (0, 0, 255)}.get(rl, (255, 255, 255))
            cv2.putText(display_frame, f"Risk: {rl} ({rc:.0%})",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Apply privacy filter
            if state.privacy_filter and state.privacy_filter.enabled:
                display_frame = state.privacy_filter.blur_faces(display_frame)

            # Apply heatmap overlay
            heatmap_raw = result.get('heatmap')
            if heatmap_raw and state.heatmap_gen:
                hm = np.array(heatmap_raw, dtype=np.uint8)
                colored = state.heatmap_gen.apply_colormap(hm, display_frame.shape[:2])
                display_frame = cv2.addWeighted(display_frame, 0.7, colored, 0.3, 0)

            _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            result['frame_b64'] = base64.b64encode(buf).decode('utf-8')

        result['status'] = 'success'
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/api/current_state", dependencies=LOGGED_IN)
async def get_current_state():
    """Get the latest analysis results."""
    frame_b64 = None
    if state.current_frame is not None:
        # Apply privacy filter before sending
        display_frame = state.current_frame.copy()
        if state.privacy_filter and state.privacy_filter.enabled:
            display_frame = state.privacy_filter.blur_faces(display_frame)
        _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        "current_frame": frame_b64,
        "people_count": len(state.current_detections),
        "detections": state.current_detections,
        "tracks": state.current_tracks,
        "features": state.current_features,
        "heatmap": state.current_heatmap,
        "risk_prediction": state.current_risk,
        "anomaly_detection": state.current_anomaly,
        "predictions": state.current_predictions,
        "control_suggestions": state.current_suggestions,
        "alerts": state.current_alerts,
        "frame_number": state.frame_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/alerts", dependencies=LOGGED_IN)
async def get_alerts():
    """Get current and recent alerts."""
    return {
        "active_alerts": state.current_alerts,
        "alert_count": len(state.current_alerts),
        "recent_history": list(state.alert_log)[-20:],
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/history", dependencies=LOGGED_IN)
async def get_history(limit: int = 100):
    """Get historical data for charts."""
    data = list(state.history)[-limit:]
    return {
        "data": data,
        "total_records": len(state.history),
        "returned": len(data)
    }


@app.get("/api/predictions", dependencies=LOGGED_IN)
async def get_predictions():
    """Get LSTM-based crowd predictions."""
    return {
        "predictions": state.current_predictions,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/control_suggestions", dependencies=LOGGED_IN)
async def get_control_suggestions():
    """Get RL-based crowd control suggestions."""
    return {
        "suggestions": state.current_suggestions,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/zones", dependencies=LOGGED_IN)
async def get_zones():
    """Live per-zone counts. Uses the zones drawn in the dashboard, or a
    3x3 grid when none have been drawn yet."""
    if state.zones.zones:
        stats = state.current_zone_stats or state.zones.analyze(state.current_detections)
        return {"zones": stats, "custom": True, "timestamp": datetime.now().isoformat()}

    zones = generate_zone_grid((480, 640), rows=3, cols=3)
    for zone in zones:
        zone_dets = [
            d for d in state.current_detections
            if zone['x1'] <= d['center'][0] < zone['x2']
            and zone['y1'] <= d['center'][1] < zone['y2']
        ]
        zone['people_count'] = len(zone_dets)
        zone['density'] = len(zone_dets) / max(
            (zone['x2'] - zone['x1']) * (zone['y2'] - zone['y1']) * 0.0001, 1
        )
        if len(zone_dets) > 10:
            zone['risk'] = 'high'
        elif len(zone_dets) > 5:
            zone['risk'] = 'medium'
        else:
            zone['risk'] = 'low'

    return {"zones": zones, "custom": False, "timestamp": datetime.now().isoformat()}


class ZonesUpdate(BaseModel):
    zones: List[Dict[str, Any]]


@app.get("/api/zones/definitions", dependencies=LOGGED_IN)
async def get_zone_definitions():
    """The saved zone shapes, types and capacities (for the Zone Editor)."""
    return {"zones": state.zones.zones}


@app.put("/api/zones", dependencies=ADMIN_ONLY)
async def save_zones(req: ZonesUpdate):
    """Replace all zones (admin only)."""
    try:
        saved = state.zones.set_zones(req.zones)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    state.current_zone_stats = state.zones.analyze(state.current_detections)
    return {"status": "saved", "zones": saved}


@app.get("/api/scenarios", dependencies=LOGGED_IN)
async def get_scenarios():
    """Get available simulation scenarios."""
    if state.simulator:
        return {"scenarios": state.simulator.get_available_scenarios()}
    return {"scenarios": {}}


@app.post("/api/simulate", dependencies=ADMIN_ONLY)
async def start_simulation(req: SimulationRequest):
    """Start a simulation scenario."""
    if not state.simulator:
        raise HTTPException(status_code=503, detail="Simulator not available")

    # Stop a simulation that is already running, otherwise two loops would
    # step the same simulator at once.
    old_task = getattr(state, 'sim_task', None)
    if old_task is not None and not old_task.done():
        state.simulation_active = False
        state.simulator.is_running = False
        try:
            await asyncio.wait_for(old_task, timeout=5)
        except Exception:
            old_task.cancel()

    # Start from a clean history so the previous video/scenario doesn't
    # leak into this one's trend features, alerts and forecasts.
    _reset_stream_state('simulation')

    state.simulation_active = True
    config = state.simulator.start_scenario(req.scenario)

    # Run simulation in background
    state.sim_task = asyncio.create_task(_run_simulation())

    return {
        "status": "started",
        "scenario": req.scenario,
        "config": {k: v for k, v in config.items() if not callable(v)},
        "message": f"Simulation '{req.scenario}' started. Connect via WebSocket for live updates."
    }


@app.post("/api/simulate/stop", dependencies=ADMIN_ONLY)
async def stop_simulation():
    """Stop the running simulation."""
    state.simulation_active = False
    if state.simulator:
        state.simulator.is_running = False
    return {"status": "stopped"}


@app.post("/api/auth/login")
async def login(req: LoginRequest, request: Request):
    """Log in with a username and password; returns a token that expires."""
    client = request.client.host if request.client else "?"
    try:
        result = state.users.login(req.username, req.password, client=client)
    except AuthError as e:
        code = 429 if "Too many" in str(e) else 401
        raise HTTPException(status_code=code, detail=str(e))
    logger.info(f"User '{result['user']['username']}' logged in from {client}")
    return {"status": "success", **result}


@app.get("/api/auth/me")
async def who_am_i(user: Dict = Depends(current_user)):
    return {"user": user}


class PasswordChange(BaseModel):
    old_password: str
    new_password: str


@app.post("/api/auth/password")
async def change_password(req: PasswordChange, request: Request, user: Dict = Depends(current_user)):
    """Change your own password (also logs out your other sessions)."""
    try:
        state.users.login(user["username"], req.old_password,
                          client=request.client.host if request.client else "?")
        state.users.set_password(user["username"], req.new_password)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "changed", "message": "Password changed. Please log in again."}


class NewUser(BaseModel):
    username: str
    password: str
    role: str = "viewer"


@app.get("/api/auth/users", dependencies=ADMIN_ONLY)
async def list_users():
    return {"users": state.users.list_users(), "roles": list(ROLES)}


@app.post("/api/auth/users", dependencies=ADMIN_ONLY)
async def create_user(req: NewUser):
    try:
        state.users.add_user(req.username, req.password, req.role)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "created", "users": state.users.list_users()}


@app.delete("/api/auth/users/{username}")
async def remove_user(username: str, user: Dict = Depends(admin_user)):
    if username == user["username"]:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")
    try:
        state.users.delete_user(username)
    except AuthError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "deleted", "users": state.users.list_users()}


class NotifySettings(BaseModel):
    settings: Dict[str, Any]


@app.get("/api/notify", dependencies=ADMIN_ONLY)
async def get_notify_settings():
    """Notification settings (secrets masked) and recent send results."""
    return state.notifier.public_config()


@app.put("/api/notify", dependencies=ADMIN_ONLY)
async def update_notify_settings(req: NotifySettings):
    try:
        return state.notifier.update(req.settings)
    except (ValueError, TypeError) as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/notify/test", dependencies=ADMIN_ONLY)
async def send_test_notification():
    """Send a test message to every configured channel and report the result."""
    if not (state.notifier.telegram_ready or state.notifier.email_ready):
        raise HTTPException(status_code=400, detail="Set up Telegram or email first")
    result = await asyncio.to_thread(state.notifier.send_test)
    return {"result": result}


@app.post("/api/notify/telegram/find_chat", dependencies=ADMIN_ONLY)
async def find_telegram_chat():
    """After you send any message to your bot, this finds your chat id."""
    try:
        chat = await asyncio.to_thread(state.notifier.find_telegram_chat_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not reach Telegram: {e}")
    if not chat:
        raise HTTPException(status_code=404, detail="No messages yet. Send any message to your bot, then try again.")
    return state.notifier.update({"telegram_chat_id": chat})


@app.post("/api/train/lstm", dependencies=ADMIN_ONLY)
async def train_lstm(req: TrainRequest):
    """Train/retrain the LSTM predictor."""
    if not state.lstm_predictor:
        raise HTTPException(status_code=503, detail="LSTM predictor not available")

    result = state.lstm_predictor.train(epochs=req.epochs, n_scenarios=req.n_scenarios)
    return {"status": "trained", "result": result}


@app.post("/api/train/rl", dependencies=ADMIN_ONLY)
async def train_rl():
    """Train/retrain the RL control agent."""
    if not state.rl_agent:
        raise HTTPException(status_code=503, detail="RL agent not available")

    result = state.rl_agent.train_simulated(n_episodes=500)
    return {"status": "trained", "result": result}


@app.get("/api/privacy/toggle", dependencies=ADMIN_ONLY)
async def toggle_privacy(enabled: Optional[bool] = None):
    """Toggle privacy filter (face blurring)."""
    if state.privacy_filter:
        state.privacy_filter.toggle(enabled)
        return {
            "enabled": state.privacy_filter.enabled,
            "stats": state.privacy_filter.get_stats()
        }
    return {"enabled": False}


@app.get("/api/logs/download", dependencies=ADMIN_ONLY)
async def download_logs():
    """Download system logs as JSON."""
    log_data = {
        "alert_history": list(state.alert_log),
        "analysis_history": list(state.history)[-200:],
        "system_info": {
            "total_frames": state.frame_count,
            "timestamp": datetime.now().isoformat()
        }
    }

    def _serialize(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return str(obj)

    content = json.dumps(log_data, default=_serialize, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=crowd_risk_logs.json"}
    )


# ─── WebSocket ───────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint for real-time updates. Connect with ?token=<login token>."""
    try:
        state.users.verify_token(ws.query_params.get("token", ""))
    except AuthError:
        await ws.close(code=1008)   # policy violation: not logged in
        return
    await ws.accept()
    state.ws_clients.append(ws)
    logger.info(f"WebSocket client connected ({len(state.ws_clients)} total)")

    try:
        while True:
            # Keep connection alive; process any client messages
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=30)
                msg = json.loads(data)

                if msg.get("type") == "ping":
                    await ws.send_json({"type": "pong"})

                elif msg.get("type") == "frame":
                    # Client sending a frame for processing
                    result = _process_frame_internal(msg["image"])
                    await ws.send_json({"type": "result", "data": result})

            except asyncio.TimeoutError:
                # Send heartbeat
                await ws.send_json({"type": "heartbeat", "timestamp": time.time()})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
        logger.info(f"WebSocket client disconnected ({len(state.ws_clients)} remaining)")


async def broadcast_ws(data: dict):
    """Send data to all connected WebSocket clients."""
    disconnected = []
    for ws in state.ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        state.ws_clients.remove(ws)


# ─── Frame Processing Core ──────────────────────────────────────
def _snapshot_jpeg(frame: np.ndarray, detections: List[Dict], count: int) -> Optional[bytes]:
    """Camera frame with people boxed, for notifications. Faces are blurred
    when the privacy filter is on."""
    try:
        img = frame.copy()
        if state.privacy_filter and state.privacy_filter.enabled:
            img = state.privacy_filter.blur_faces(img)
        for d in detections[:200]:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for z in state.current_zone_stats:
            pts = np.array(z['points'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, f"{count} people  {datetime.now():%H:%M:%S}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    except Exception as e:
        logger.warning(f"Snapshot failed: {e}")
        return None


def _reset_stream_state(source: str):
    """
    Forget all per-stream history: trends, tracks, motion baseline, forecast
    window, count smoothing. Called when a simulation starts and whenever the
    input switches between simulation and camera/video, because e.g. the
    panic detector's idea of "normal speed" from drawn simulation people
    makes real shoppers look like they are running.
    """
    state.analyzer.reset()
    if state.tracker is not None:
        state.tracker = PersonTracker(use_deep_sort=state.tracker.use_deep_sort, max_disappeared=30)
    if state.panic_detector is not None:
        state.panic_detector.reset()
    if state.evacuation_detector is not None:
        state.evacuation_detector.reset()
    state.stream_time = 0.0
    if state.lstm_predictor is not None:
        state.lstm_predictor._history.clear()
    cc = getattr(state.detector, 'count_corrector', None) if state.detector else None
    if cc is not None and hasattr(cc, 'reset'):
        cc.reset()
    state.stream_source = source

def _process_frame_internal(image_b64: str) -> Dict:
    """Core frame processing pipeline."""
    start = time.time()

    # Decode image
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")

    if frame.shape[:2] != (480, 640):
        frame = cv2.resize(frame, (640, 480))

    return _process_raw_frame(frame)


def _process_raw_frame(frame: np.ndarray, sim_detections=None, frames_elapsed: float = 1.0) -> Dict:
    """Process a raw numpy frame through the full pipeline."""
    start = time.time()
    state.frame_count += 1

    source = 'simulation' if sim_detections is not None else 'camera'
    if getattr(state, 'stream_source', None) != source:
        _reset_stream_state(source)

    # 1. Detect people
    if sim_detections is not None:
        detections = sim_detections
        count = len(detections)
    else:
        detections, count = state.detector.detect_people(frame)

    # 2. Track people
    tracks = {}
    tracking_summary = {}
    if state.tracker:
        tracks = state.tracker.update(detections, frame=frame)
        tracking_summary = state.tracker.get_tracking_summary()

    # 2b. Panic detection from movement (works for camera, video and simulation)
    panic = {}
    if state.panic_detector is not None and state.tracker:
        panic = state.panic_detector.update(tracks, frames_elapsed=frames_elapsed)
    state.current_panic = panic

    # 2c. Evacuation detection from the count. Time in seconds: simulation
    # time for simulations, wall clock for a camera or video.
    if sim_detections is not None:
        sim_fps = getattr(state.simulator, 'fps', 15) or 15
        state.stream_time += frames_elapsed / sim_fps
        t_now = state.stream_time
    else:
        t_now = time.time()
    evacuation = {}
    if state.evacuation_detector is not None:
        evacuation = state.evacuation_detector.update(count, t_now)

    # 2d. Per-zone counts for the zones drawn in the dashboard
    zone_stats = state.zones.analyze(detections, total_count=count)
    state.current_zone_stats = zone_stats

    # 3. Analyze crowd features (count may be calibrated by the count corrector)
    features = state.analyzer.update_frame(frame, detections, people_count=count)

    # 4. Generate heatmap
    heatmap = None
    if state.heatmap_gen:
        heatmap_raw = state.heatmap_gen.generate_heatmap(detections, frame.shape[:2])
        if heatmap_raw is not None and np.max(heatmap_raw) > 0:
            heatmap = heatmap_raw.tolist()

    # The risk and anomaly models were trained on the real Mall camera.
    # Simulated frames look nothing like it (every frame came out as an
    # "anomaly"), so simulations use the count-based rules instead.
    real_camera = sim_detections is None

    # 5. Risk prediction
    risk_result = {}
    if real_camera and state.risk_predictor and state.risk_predictor.is_trained:
        fv = state.analyzer.get_feature_vector(state.risk_predictor.feature_names)
        if len(fv) > 0:
            risk_level, confidence = state.risk_predictor.predict(fv)
            risk_result = {'risk_level': risk_level, 'confidence': float(confidence)}

    # 6. Anomaly detection
    anomaly_result = {}
    # Skip the first frames: trend features need history, and an empty
    # history looks "anomalous" to a model trained on a running video.
    warmed_up = len(state.analyzer.count_history) >= 10
    if real_camera and state.anomaly_detector and state.anomaly_detector.is_trained and warmed_up:
        fv = state.analyzer.get_feature_vector(state.anomaly_detector.feature_names)
        if len(fv) > 0:
            is_anomaly, score, details = state.anomaly_detector.detect_anomaly(fv)
            anomaly_result = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score),
            }

    # 7. LSTM predictions
    predictions = {}
    if state.lstm_predictor:
        density = features.get('density_per_frame', 0)
        avg_speed = features.get('avg_magnitude', 0)
        flow_mag = features.get('movement_concentration', 0)
        state.lstm_predictor.add_observation(count, density, avg_speed, flow_mag)
        predictions = state.lstm_predictor.predict()

    # 8. RL control suggestions
    suggestions = []
    if state.rl_agent and state.rl_agent.is_trained:
        density_norm = min(count / 50.0, 1.0)
        avg_spd = tracking_summary.get('average_speed', 0)
        risk_lvl = risk_result.get('risk_level', 'low')
        conflict = tracking_summary.get('direction_conflict', False)
        suggestions = state.rl_agent.get_suggestions(
            density=density_norm, avg_speed=avg_spd,
            risk_level=risk_lvl, direction_conflict=conflict
        )

    # Rule-based risk fallback when no model is trained.
    # Aligned with the capacity alerts so the dashboard badge and the
    # warnings always agree (medium >= 80% capacity, high > capacity).
    if not risk_result:
        cap = state.alert_system.capacity_limit if state.alert_system else 30
        ratio = state.alert_system.alert_ratio if state.alert_system else 0.8
        if count > cap:
            risk_result = {'risk_level': 'high', 'confidence': 0.9}
        elif count >= cap * ratio:
            risk_result = {'risk_level': 'medium', 'confidence': 0.7}
        else:
            risk_result = {'risk_level': 'low', 'confidence': 0.8}

    # Panic or a rapid evacuation overrides the risk level: both point to an
    # incident, whatever the crowd size
    if evacuation.get('is_evacuation') and not panic.get('is_panic'):
        risk_result = {**risk_result, 'risk_level': 'high',
                       'confidence': max(float(risk_result.get('confidence', 0)), 0.85),
                       'reason': 'evacuation'}
    if panic.get('is_panic'):
        risk_result = {**risk_result, 'risk_level': 'high',
                       'confidence': max(float(risk_result.get('confidence', 0)), 0.9),
                       'reason': 'panic'}

    # 9. Generate alerts
    alerts = []
    if state.alert_system:
        raw_alerts = state.alert_system.evaluate_risk(risk_result, anomaly_result, {**features, 'panic': panic, 'evacuation': evacuation})
        raw_alerts += zone_alerts(zone_stats, time.time())
        # Copy before making serializable: AlertSystem keeps the originals in
        # active_alerts and later does `now - alert['timestamp']`, which
        # crashed with "float - str" once the timestamp became a string.
        alerts = [{**a, 'timestamp': datetime.now().isoformat()} for a in raw_alerts]
        state.alert_log.extend(alerts)

        # Phone / email notifications (cooldown per alert kind, sent in background)
        to_send = state.notifier.wants(alerts)
        if to_send:
            state.notifier.notify(to_send, _snapshot_jpeg(frame, detections, count),
                                  context='simulation' if sim_detections is not None else 'live camera')

    proc_time = time.time() - start
    fps = 1.0 / proc_time if proc_time > 0 else 0

    if state.perf_monitor:
        state.perf_monitor.update(proc_time, fps)

    # Make features serializable
    safe_features = {}
    for k, v in features.items():
        if isinstance(v, np.ndarray):
            continue  # skip density_map etc.
        elif isinstance(v, (np.floating, np.integer)):
            safe_features[k] = float(v)
        elif isinstance(v, (np.bool_,)):
            safe_features[k] = bool(v)
        else:
            safe_features[k] = v

    # Update global state
    state.current_frame = frame
    state.current_detections = detections
    state.current_tracks = {str(k): v for k, v in tracks.items()}
    state.current_features = safe_features
    state.current_risk = risk_result
    state.current_anomaly = anomaly_result
    state.current_predictions = predictions
    state.current_suggestions = suggestions
    state.current_alerts = alerts
    state.current_heatmap = heatmap

    # Add to history
    history_entry = {
        'frame': state.frame_count,
        'timestamp': datetime.now().isoformat(),
        'people_count': count,
        'density': safe_features.get('density_per_frame', 0),
        'risk_level': risk_result.get('risk_level', 'low'),
        'risk_confidence': risk_result.get('confidence', 0),
        'is_anomaly': anomaly_result.get('is_anomaly', False),
        'avg_speed': tracking_summary.get('average_speed', 0),
        'alert_count': len(alerts),
        'fps': round(fps, 1)
    }
    state.history.append(history_entry)

    result = {
        'frame_number': state.frame_count,
        'people_count': count,
        'detections': detections[:50],  # Cap for bandwidth
        'tracks': {str(k): v for k, v in list(tracks.items())[:50]},
        'features': safe_features,
        'heatmap': heatmap,
        'risk_prediction': risk_result,
        'anomaly_detection': anomaly_result,
        'predictions': predictions,
        'control_suggestions': suggestions,
        'alerts': alerts,
        'panic': panic,
        'evacuation': evacuation,
        'zones': zone_stats,
        'tracking_summary': {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in tracking_summary.items()
        },
        'processing_time': round(proc_time, 4),
        'fps': round(fps, 1),
        'timestamp': datetime.now().isoformat()
    }

    return result


# ─── Simulation Runner ───────────────────────────────────────────
async def _run_simulation():
    """Run simulation in background and broadcast results via WebSocket."""
    logger.info("Simulation background task started")
    import time
    target_fps = 15
    last_time = time.time()

    while state.simulation_active and state.simulator and state.simulator.is_running:
        current_time = time.time()
        elapsed = current_time - last_time
        steps = max(1, int(elapsed * target_fps))
        last_time = current_time

        data = state.simulator.step(steps=steps)
        if not data:
            break

        frame = data.get('frame')
        if frame is not None:
            # Process through full pipeline
            result = _process_raw_frame(frame, sim_detections=data.get('detections'),
                                        frames_elapsed=steps)

            # Add simulation-specific info
            result['simulation'] = {
                'scenario': data.get('scenario'),
                'progress': data.get('progress', 0),
                'is_running': data.get('is_running', False),
                'sim_features': data.get('features', {})
            }

            # Encode frame for WebSocket
            display_frame = frame.copy()
            if state.privacy_filter and state.privacy_filter.enabled:
                display_frame = state.privacy_filter.blur_faces(display_frame)
            _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
            result['frame_b64'] = base64.b64encode(buf).decode('utf-8')

            # Broadcast to all WebSocket clients
            await broadcast_ws({"type": "simulation_frame", "data": result})

        # Control frame rate
        sleep_time = max(0, (1.0 / target_fps) - (time.time() - last_time))
        await asyncio.sleep(sleep_time)

    state.simulation_active = False
    await broadcast_ws({"type": "simulation_complete"})
    logger.info("Simulation complete")


# ─── Run ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
