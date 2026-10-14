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
import threading
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
from cameras import (CameraStream, CameraReader, CameraConfig, zones_path_for,
                     mask_source, BUILTIN)
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
    """The shared models, plus one CameraStream per camera (src/cameras.py)."""

    def __init__(self):
        # Shared models, loaded once
        self.detector: Optional[PeopleDetector] = None
        self.risk_predictor: Optional[RiskPredictor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.lstm_predictor: Optional[LSTMCrowdPredictor] = None
        self.rl_agent: Optional[CrowdControlAgent] = None
        self.privacy_filter: Optional[PrivacyFilter] = None
        self.simulator: Optional[CrowdSimulator] = None
        self.heatmap_gen: Optional[HeatmapGenerator] = None
        self.perf_monitor: Optional[PerformanceMonitor] = None
        # Settings every camera's tracker and alert system copy
        self.tracker_template: Optional[PersonTracker] = None
        self.alert_template: Optional[AlertSystem] = None

        self.initialized = False
        self.processing = False
        self.simulation_active = False
        self.frame_count = 0   # all cameras together

        # WebSocket clients and the camera each one is watching
        self.ws_clients: List[WebSocket] = []
        self.ws_watch: Dict[int, str] = {}
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        data_dir = os.environ.get("CROWDGUARD_DATA_DIR",
                                  os.path.join(os.path.dirname(__file__), '..', 'data'))
        self.data_dir = data_dir
        # Accounts and login tokens (see src/auth.py)
        self.users = UserStore(data_dir)
        # Telegram / email notifications for critical alerts (see src/notifier.py)
        self.notifier = Notifier(os.path.join(data_dir, 'notify.json'))

        # Cameras (see src/cameras.py). YOLO and the models are shared, so
        # one frame is analysed at a time.
        self.camera_config = CameraConfig(os.path.join(data_dir, 'cameras.json'))
        self.cameras: Dict[str, CameraStream] = {}
        self.readers: Dict[str, CameraReader] = {}
        self.pipeline_lock = threading.Lock()

    def camera(self, cam_id: str = 'main') -> CameraStream:
        """The stream state of a camera (created on first use)."""
        cam_id = cam_id or 'main'
        cam = self.cameras.get(cam_id)
        if cam is not None:
            return cam
        if cam_id == 'main':
            name, kind = 'This browser', 'browser'
        elif cam_id == 'simulation':
            name, kind = 'Simulation', 'simulation'
        else:
            cfg = self.camera_config.get(cam_id)
            if cfg is None:
                raise KeyError(cam_id)
            name, kind = cfg['name'], 'server'
        cam = CameraStream(cam_id, name, kind, self, zones_path_for(self.data_dir, cam_id))
        self.cameras[cam_id] = cam
        return cam

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

        # ML models
        state.risk_predictor = RiskPredictor(model_type='random_forest')
        state.anomaly_detector = AnomalyDetector(method='isolation_forest', contamination=0.1)

        # Multi-object tracker (each camera gets its own with these settings)
        state.tracker_template = PersonTracker(use_deep_sort=True, max_disappeared=30)

        # LSTM predictor
        state.lstm_predictor = LSTMCrowdPredictor(window_size=30, prediction_steps=10)

        # RL agent
        state.rl_agent = CrowdControlAgent()
        state.rl_agent.train_simulated(n_episodes=200)

        # Privacy filter
        state.privacy_filter = PrivacyFilter(blur_strength=51, enabled=True)

        # Simulator
        state.simulator = CrowdSimulator(frame_width=640, frame_height=480, fps=15)

        # Alert system settings (each camera gets its own copy)
        state.alert_template = AlertSystem(
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

        # Server cameras saved in data/cameras.json, and the loop that analyses them
        state.loop = asyncio.get_running_loop()
        for cfg in state.camera_config.cameras:
            if cfg.get('enabled', True):
                _start_reader(cfg)
        threading.Thread(target=_camera_loop, daemon=True, name='camera-loop').start()
        if state.camera_config.cameras:
            logger.info(f"Cameras: {', '.join(c['name'] for c in state.camera_config.cameras)}")

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
            "cameras": len(state.camera_config.cameras),
            "risk_predictor": state.risk_predictor is not None,
            "anomaly_detector": state.anomaly_detector is not None,
            "tracker": state.tracker_template is not None,
            "lstm_predictor": state.lstm_predictor is not None,
            "rl_agent": state.rl_agent is not None,
            "privacy_filter": state.privacy_filter is not None,
            "simulator": state.simulator is not None,
        },
        "performance": state.perf_monitor.get_stats() if state.perf_monitor else {},
        "timestamp": datetime.now().isoformat()
    }


def _cam(camera: str) -> CameraStream:
    """The camera named in ?camera= (default: this browser's feed)."""
    try:
        return state.camera(camera)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera '{camera}'")


@app.post("/api/process_frame", dependencies=LOGGED_IN)
async def process_frame(data: FrameInput, camera: str = 'main'):
    """Process a single base64-encoded frame from the browser and return the
    full AI analysis. Only the browser camera ('main') takes frames this way;
    server cameras are read by the backend itself."""
    if not state.initialized:
        raise HTTPException(status_code=503, detail="System not initialized")
    if camera in state.readers:
        raise HTTPException(status_code=400, detail="This camera is read by the server; frames cannot be sent to it")

    try:
        result = await asyncio.to_thread(_process_frame_internal, data.image, camera)
        cam = state.camera(camera)
        # The processed view (boxes, privacy blur, heatmap) for the dashboard
        if data.return_frame and cam.current_frame is not None:
            result['frame_b64'] = base64.b64encode(_annotated_jpeg(cam.current_frame, result)).decode('utf-8')
        result['status'] = 'success'
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/current_state", dependencies=LOGGED_IN)
async def get_current_state(camera: str = 'main'):
    """Get the latest analysis results of one camera."""
    cam = _cam(camera)
    frame_b64 = None
    if cam.last_annotated_jpeg is not None and cam.kind != 'browser':
        frame_b64 = base64.b64encode(cam.last_annotated_jpeg).decode('utf-8')
    elif cam.current_frame is not None:
        # Apply privacy filter before sending
        display_frame = cam.current_frame.copy()
        if state.privacy_filter and state.privacy_filter.enabled:
            display_frame = state.privacy_filter.blur_faces(display_frame)
        _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_b64 = base64.b64encode(buf).decode('utf-8')

    return {
        "camera": cam.id,
        "current_frame": frame_b64,
        "people_count": cam.current_count,
        "detections": cam.current_detections,
        "tracks": cam.current_tracks,
        "features": cam.current_features,
        "heatmap": cam.current_heatmap,
        "risk_prediction": cam.current_risk,
        "anomaly_detection": cam.current_anomaly,
        "predictions": cam.current_predictions,
        "control_suggestions": cam.current_suggestions,
        "alerts": cam.current_alerts,
        "frame_number": cam.frame_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/alerts", dependencies=LOGGED_IN)
async def get_alerts(camera: str = 'main'):
    """Get current and recent alerts of one camera ('all' = every camera)."""
    if camera == 'all':
        active = [a for c in state.cameras.values() for a in c.current_alerts]
        recent = sorted((a for c in state.cameras.values() for a in c.alert_log),
                        key=lambda a: a.get('timestamp', ''))[-50:]
    else:
        cam = _cam(camera)
        active, recent = cam.current_alerts, list(cam.alert_log)[-20:]
    return {
        "active_alerts": active,
        "alert_count": len(active),
        "recent_history": recent,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/history", dependencies=LOGGED_IN)
async def get_history(limit: int = 100, camera: str = 'main'):
    """Get historical data for charts."""
    cam = _cam(camera)
    data = list(cam.history)[-limit:]
    return {
        "data": data,
        "total_records": len(cam.history),
        "returned": len(data)
    }


@app.get("/api/predictions", dependencies=LOGGED_IN)
async def get_predictions(camera: str = 'main'):
    """Get LSTM-based crowd predictions."""
    return {
        "predictions": _cam(camera).current_predictions,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/control_suggestions", dependencies=LOGGED_IN)
async def get_control_suggestions(camera: str = 'main'):
    """Get RL-based crowd control suggestions."""
    return {
        "suggestions": _cam(camera).current_suggestions,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/api/zones", dependencies=LOGGED_IN)
async def get_zones(camera: str = 'main'):
    """Live per-zone counts. Uses the zones drawn in the dashboard, or a
    3x3 grid when none have been drawn yet."""
    cam = _cam(camera)
    if cam.zones.zones:
        stats = cam.current_zone_stats or cam.zones.analyze(cam.current_detections)
        return {"zones": stats, "custom": True, "timestamp": datetime.now().isoformat()}

    zones = generate_zone_grid((480, 640), rows=3, cols=3)
    for zone in zones:
        zone_dets = [
            d for d in cam.current_detections
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
async def get_zone_definitions(camera: str = 'main'):
    """The saved zone shapes, types and capacities (for the Zone Editor)."""
    return {"zones": _cam(camera).zones.zones}


@app.put("/api/zones", dependencies=ADMIN_ONLY)
async def save_zones(req: ZonesUpdate, camera: str = 'main'):
    """Replace all zones of one camera (admin only)."""
    cam = _cam(camera)
    try:
        saved = cam.zones.set_zones(req.zones)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    cam.current_zone_stats = cam.zones.analyze(cam.current_detections)
    return {"status": "saved", "zones": saved}


# ─── Cameras ─────────────────────────────────────────────────────
class CameraIn(BaseModel):
    name: str
    source: str


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    source: Optional[str] = None
    enabled: Optional[bool] = None


def _thumb_b64(cam: CameraStream) -> Optional[str]:
    """A small picture of the camera's latest analysed frame."""
    try:
        if cam.last_annotated_jpeg is not None:
            img = cv2.imdecode(np.frombuffer(cam.last_annotated_jpeg, np.uint8), cv2.IMREAD_COLOR)
        elif cam.current_frame is not None:
            img = cam.current_frame
            if state.privacy_filter and state.privacy_filter.enabled:
                img = state.privacy_filter.blur_faces(img.copy())
        else:
            return None
        img = cv2.resize(img, (320, 240))
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 60])
        return base64.b64encode(buf).decode('utf-8') if ok else None
    except Exception:
        return None


def _camera_info(cam_id: str, admin: bool, thumbs: bool) -> Dict:
    cfg = state.camera_config.get(cam_id)
    cam = state.cameras.get(cam_id)
    if cam is None and cfg is None and cam_id not in BUILTIN:
        raise KeyError(cam_id)
    if cam is not None:
        info = cam.summary()
    else:
        name = cfg['name'] if cfg else ('This browser' if cam_id == 'main' else 'Simulation')
        kind = 'server' if cfg else ('browser' if cam_id == 'main' else 'simulation')
        info = {"id": cam_id, "name": name, "kind": kind, "people_count": 0, "risk_level": "low",
                "alert_count": 0, "critical": False, "panic": False, "zones": 0, "frames": 0, "last_update": 0}
    reader = state.readers.get(cam_id)
    if cfg is not None:
        info["enabled"] = cfg.get("enabled", True)
        info["status"] = reader.status if reader else "off"
        info["error"] = reader.error if reader else ""
        info["source_fps"] = round(reader.fps, 1) if reader else 0
        if admin:
            info["source"] = mask_source(cfg["source"])
    else:
        recent = time.time() - info["last_update"] < 5
        if cam_id == 'simulation':
            info["status"] = "live" if state.simulation_active else "off"
        else:
            info["status"] = "live" if recent else "idle"
    if thumbs and cam is not None:
        info["thumbnail"] = _thumb_b64(cam)
    return info


@app.get("/api/cameras")
async def list_cameras(thumbs: bool = True, user: Dict = Depends(current_user)):
    """Every camera with its latest count, risk, alerts and a thumbnail."""
    admin = user.get("role") == "admin"
    ids = ['main'] + [c['id'] for c in state.camera_config.cameras]
    if state.simulation_active or 'simulation' in state.cameras:
        ids.append('simulation')
    return {"cameras": [_camera_info(i, admin, thumbs) for i in ids],
            "max_fps_per_camera": CAMERA_MAX_FPS}


@app.post("/api/cameras", dependencies=ADMIN_ONLY)
async def add_camera(req: CameraIn):
    """Add a camera the server reads itself: a webcam number (0, 1, ...),
    an rtsp:// or http:// stream, or a video file path."""
    try:
        cfg = state.camera_config.add(req.name, req.source)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _start_reader(cfg)
    logger.info(f"Camera added: {cfg['name']} ({mask_source(cfg['source'])})")
    return _camera_info(cfg['id'], True, False)


@app.put("/api/cameras/{cam_id}", dependencies=ADMIN_ONLY)
async def update_camera(cam_id: str, req: CameraUpdate):
    """Rename a camera, change its source, or turn it on or off."""
    try:
        before = dict(state.camera_config.get(cam_id) or {})
        cfg = state.camera_config.update(cam_id, req.dict(exclude_none=True))
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera '{cam_id}'")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if cam_id in state.cameras:
        state.cameras[cam_id].name = cfg['name']
    if not cfg.get('enabled', True):
        _stop_reader(cam_id)
    elif cfg['source'] != before.get('source') or cam_id not in state.readers:
        _start_reader(cfg)
        if cam_id in state.cameras:
            state.cameras[cam_id].reset('camera')
    return _camera_info(cam_id, True, False)


@app.delete("/api/cameras/{cam_id}", dependencies=ADMIN_ONLY)
async def delete_camera(cam_id: str):
    """Remove a server camera (its zones file is kept)."""
    if cam_id in BUILTIN:
        raise HTTPException(status_code=400, detail="This camera is built in and cannot be removed")
    try:
        state.camera_config.remove(cam_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No camera '{cam_id}'")
    _stop_reader(cam_id)
    with state.pipeline_lock:
        state.cameras.pop(cam_id, None)
    return {"status": "deleted", "id": cam_id}


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
    _reset_stream_state('simulation', 'simulation')

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

    result = await asyncio.to_thread(state.lstm_predictor.train, epochs=req.epochs, n_scenarios=req.n_scenarios)
    with state.pipeline_lock:
        for cam in state.cameras.values():
            cam.refresh_models()
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
    """Download system logs as JSON (every camera)."""
    log_data = {
        "cameras": {
            cam.id: {
                "name": cam.name,
                "alert_history": list(cam.alert_log),
                "analysis_history": list(cam.history)[-200:],
                "frames": cam.frame_count,
            } for cam in state.cameras.values()
        },
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

                elif msg.get("type") == "watch":
                    # Which camera this dashboard shows (live frames are only
                    # sent for that one)
                    state.ws_watch[id(ws)] = str(msg.get("camera") or "main")

                elif msg.get("type") == "frame":
                    # Client sending a frame for processing
                    result = await asyncio.to_thread(_process_frame_internal, msg["image"],
                                                     str(msg.get("camera") or "main"))
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
        state.ws_watch.pop(id(ws), None)
        logger.info(f"WebSocket client disconnected ({len(state.ws_clients)} remaining)")


async def broadcast_ws(data: dict, camera: Optional[str] = None):
    """Send data to all connected WebSocket clients (or only to the ones
    watching `camera`)."""
    disconnected = []
    for ws in list(state.ws_clients):
        if camera is not None and state.ws_watch.get(id(ws), 'main') != camera:
            continue
        try:
            await ws.send_json(data)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in state.ws_clients:
            state.ws_clients.remove(ws)
        state.ws_watch.pop(id(ws), None)


# ─── Frame Processing Core ──────────────────────────────────────
def _snapshot_jpeg(cam: CameraStream, frame: np.ndarray, detections: List[Dict], count: int) -> Optional[bytes]:
    """Camera frame with people boxed, for notifications. Faces are blurred
    when the privacy filter is on."""
    try:
        img = frame.copy()
        if state.privacy_filter and state.privacy_filter.enabled:
            img = state.privacy_filter.blur_faces(img)
        for d in detections[:200]:
            x1, y1, x2, y2 = [int(v) for v in d['bbox']]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for z in cam.current_zone_stats:
            pts = np.array(z['points'], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(img, [pts], True, (0, 200, 255), 2)
        cv2.rectangle(img, (0, 0), (img.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(img, f"{cam.name}: {count} people  {datetime.now():%H:%M:%S}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else None
    except Exception as e:
        logger.warning(f"Snapshot failed: {e}")
        return None


def _annotated_jpeg(frame: np.ndarray, result: Dict, quality: int = 70) -> bytes:
    """The frame with boxes, count and risk drawn on it, privacy filter and
    heatmap applied: what the dashboard shows."""
    display_frame = frame.copy()
    if state.privacy_filter and state.privacy_filter.enabled:
        display_frame = state.privacy_filter.blur_faces(display_frame)
    for det in result.get('detections', [])[:50]:
        x1, y1, x2, y2 = [int(v) for v in det.get('bbox', [0, 0, 0, 0])]
        conf = det.get('confidence', 0)
        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display_frame, f"{conf:.0%}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    cv2.putText(display_frame, f"People: {result.get('people_count', 0)}",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    risk = result.get('risk_prediction', {})
    rl = risk.get('risk_level', 'low').upper()
    rc = risk.get('confidence', 0)
    color = {'LOW': (0, 255, 0), 'MEDIUM': (0, 255, 255), 'HIGH': (0, 0, 255)}.get(rl, (255, 255, 255))
    cv2.putText(display_frame, f"Risk: {rl} ({rc:.0%})",
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    heatmap_raw = result.get('heatmap')
    if heatmap_raw and state.heatmap_gen:
        hm = np.array(heatmap_raw, dtype=np.uint8)
        colored = state.heatmap_gen.apply_colormap(hm, display_frame.shape[:2])
        display_frame = cv2.addWeighted(display_frame, 0.7, colored, 0.3, 0)
    _, buf = cv2.imencode('.jpg', display_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def _reset_stream_state(source: str, cam_id: str = 'main'):
    """
    Forget one camera's stream history: trends, tracks, motion baseline,
    forecast window, count smoothing. Called when a simulation starts and
    whenever a camera's input switches between simulation and video,
    because e.g. the panic detector's idea of "normal speed" from drawn
    simulation people makes real shoppers look like they are running.
    """
    state.camera(cam_id).reset(source)


def _process_frame_internal(image_b64: str, cam_id: str = 'main') -> Dict:
    """Decode a base64 frame from the browser and run the pipeline."""
    img_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image")

    if frame.shape[:2] != (480, 640):
        frame = cv2.resize(frame, (640, 480))

    return _process_raw_frame(frame, cam=state.camera(cam_id))


def _process_raw_frame(frame: np.ndarray, sim_detections=None, frames_elapsed: float = 1.0,
                       cam: Optional[CameraStream] = None) -> Dict:
    """Run one frame of one camera through the full pipeline. The shared
    models handle one frame at a time, so this holds the pipeline lock."""
    with state.pipeline_lock:
        return _process_locked(frame, sim_detections, frames_elapsed,
                               cam if cam is not None else state.camera('main'))


def _process_locked(frame: np.ndarray, sim_detections, frames_elapsed: float, cam: CameraStream) -> Dict:
    start = time.time()
    state.frame_count += 1
    cam.frame_count += 1

    source = 'simulation' if sim_detections is not None else 'camera'
    if cam.stream_source != source:
        cam.reset(source)

    # 1. Detect people (with this camera's own count smoothing)
    if sim_detections is not None:
        detections = sim_detections
        count = len(detections)
    else:
        state.detector.count_corrector = cam.count_corrector
        detections, count = state.detector.detect_people(frame)

    # 2. Track people
    tracks = cam.tracker.update(detections, frame=frame)
    tracking_summary = cam.tracker.get_tracking_summary()

    # 2b. Panic detection from movement (works for camera, video and simulation)
    panic = cam.panic_detector.update(tracks, frames_elapsed=frames_elapsed)
    cam.current_panic = panic

    # 2c. Evacuation detection from the count. Time in seconds: simulation
    # time for simulations, wall clock for a camera or video.
    if sim_detections is not None:
        sim_fps = getattr(state.simulator, 'fps', 15) or 15
        cam.stream_time += frames_elapsed / sim_fps
        t_now = cam.stream_time
    else:
        t_now = time.time()
    evacuation = cam.evacuation_detector.update(count, t_now)

    # 2d. Per-zone counts for the zones drawn for this camera
    zone_stats = cam.zones.analyze(detections, total_count=count)
    cam.current_zone_stats = zone_stats

    # 3. Analyze crowd features (count may be calibrated by the count corrector)
    features = cam.analyzer.update_frame(frame, detections, people_count=count)

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
        fv = cam.analyzer.get_feature_vector(state.risk_predictor.feature_names)
        if len(fv) > 0:
            risk_level, confidence = state.risk_predictor.predict(fv)
            risk_result = {'risk_level': risk_level, 'confidence': float(confidence)}

    # 6. Anomaly detection
    anomaly_result = {}
    # Skip the first frames: trend features need history, and an empty
    # history looks "anomalous" to a model trained on a running video.
    warmed_up = len(cam.analyzer.count_history) >= 10
    if real_camera and state.anomaly_detector and state.anomaly_detector.is_trained and warmed_up:
        fv = cam.analyzer.get_feature_vector(state.anomaly_detector.feature_names)
        if len(fv) > 0:
            is_anomaly, score, details = state.anomaly_detector.detect_anomaly(fv)
            anomaly_result = {
                'is_anomaly': bool(is_anomaly),
                'anomaly_score': float(score),
            }

    # 7. LSTM predictions
    predictions = {}
    if cam.lstm_predictor:
        density = features.get('density_per_frame', 0)
        avg_speed = features.get('avg_magnitude', 0)
        flow_mag = features.get('movement_concentration', 0)
        cam.lstm_predictor.add_observation(count, density, avg_speed, flow_mag)
        predictions = cam.lstm_predictor.predict()

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
        cap = cam.alert_system.capacity_limit
        ratio = cam.alert_system.alert_ratio
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
    raw_alerts = cam.alert_system.evaluate_risk(risk_result, anomaly_result, {**features, 'panic': panic, 'evacuation': evacuation})
    raw_alerts += zone_alerts(zone_stats, time.time())
    # Copy before making serializable: AlertSystem keeps the originals in
    # active_alerts and later does `now - alert['timestamp']`, which
    # crashed with "float - str" once the timestamp became a string.
    alerts = [{**a, 'timestamp': datetime.now().isoformat(), 'camera': cam.id, 'camera_name': cam.name}
              for a in raw_alerts]
    cam.alert_log.extend(alerts)

    # Phone / email notifications (cooldown per alert kind and camera, sent in background)
    to_send = state.notifier.wants(alerts)
    if to_send:
        kind = 'simulation' if sim_detections is not None else 'live camera'
        state.notifier.notify(to_send, _snapshot_jpeg(cam, frame, detections, count), context=kind)

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

    # Update this camera's state
    cam.current_frame = frame
    cam.current_count = count
    cam.current_detections = detections
    cam.current_tracks = {str(k): v for k, v in tracks.items()}
    cam.current_features = safe_features
    cam.current_risk = risk_result
    cam.current_anomaly = anomaly_result
    cam.current_predictions = predictions
    cam.current_suggestions = suggestions
    cam.current_alerts = alerts
    cam.current_heatmap = heatmap
    cam.last_update = time.time()

    # Add to history
    history_entry = {
        'frame': cam.frame_count,
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
    cam.history.append(history_entry)

    result = {
        'camera': cam.id,
        'camera_name': cam.name,
        'frame_number': cam.frame_count,
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


# ─── Server cameras ──────────────────────────────────────────────
CAMERA_MAX_FPS = float(os.environ.get("CROWDGUARD_CAMERA_FPS", "5"))   # analysed frames per camera per second


def _start_reader(cfg: Dict):
    old = state.readers.pop(cfg['id'], None)
    if old is not None:
        old.stop()
    state.readers[cfg['id']] = CameraReader(cfg['id'], cfg['source'])


def _stop_reader(cam_id: str):
    reader = state.readers.pop(cam_id, None)
    if reader is not None:
        reader.stop()


def _camera_loop():
    """
    Background thread: analyse the newest frame of each server camera in
    turn, fairly (the camera waiting longest goes first), at most
    CAMERA_MAX_FPS frames per camera per second. Results go to the
    dashboards watching that camera over the WebSocket.
    """
    last_seq: Dict[str, int] = {}
    last_done: Dict[str, float] = {}
    while True:
        try:
            now = time.time()
            ready = []
            for cam_id, reader in list(state.readers.items()):
                seq, frame = reader.latest()
                if frame is None or seq == last_seq.get(cam_id):
                    continue
                if now - last_done.get(cam_id, 0) < 1.0 / CAMERA_MAX_FPS:
                    continue
                ready.append((last_done.get(cam_id, 0), cam_id, seq, frame))
            if not ready or not state.initialized:
                time.sleep(0.02)
                continue
            _, cam_id, seq, frame = min(ready, key=lambda r: r[0])
            last_seq[cam_id], last_done[cam_id] = seq, time.time()
            try:
                cam = state.camera(cam_id)
            except KeyError:
                continue
            result = _process_raw_frame(frame, cam=cam)
            jpeg = _annotated_jpeg(frame, result)
            cam.last_annotated_jpeg = jpeg
            if any(w == cam_id for w in state.ws_watch.values()) and state.loop is not None:
                result['frame_b64'] = base64.b64encode(jpeg).decode('utf-8')
                asyncio.run_coroutine_threadsafe(
                    broadcast_ws({"type": "camera_frame", "camera": cam_id, "data": result}, camera=cam_id),
                    state.loop)
        except Exception as e:
            logger.error(f"Camera loop error: {e}", exc_info=True)
            time.sleep(0.5)


# ─── Simulation Runner ───────────────────────────────────────────
async def _run_simulation():
    """Run simulation in background and broadcast results via WebSocket."""
    logger.info("Simulation background task started")
    target_fps = 15
    last_time = time.time()
    cam = state.camera('simulation')

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
            # Process through full pipeline (in a thread, so the server
            # keeps answering while a camera frame is being analysed)
            result = await asyncio.to_thread(
                _process_raw_frame, frame, data.get('detections'), steps, cam)

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
            cam.last_annotated_jpeg = buf.tobytes()
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
