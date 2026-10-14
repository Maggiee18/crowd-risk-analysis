"""
Multiple cameras.

Every camera has its own stream state (count history, tracks, panic and
evacuation baselines, zones, alerts, forecast window), so two cameras never
mix their numbers. The heavy models (YOLO, risk, anomaly, LSTM weights) are
loaded once and shared.

Kinds of camera:
  main        the video or webcam the dashboard sends from your browser
              (Live Feed page), exactly as before
  simulation  the built in crowd simulator
  server      a camera the backend reads itself with OpenCV:
                0, 1, ...                    a webcam on this computer
                rtsp://user:pass@ip/stream   most CCTV / IP cameras
                http://phone-ip:8080/video   a phone running an IP camera app
                C:\\videos\\gate.mp4           a video file (loops)

Server cameras are saved in data/cameras.json (git ignored, it can hold
camera passwords) and start again when the backend starts.
"""
import copy
import json
import logging
import os
import re
import threading
import time
from collections import deque
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FRAME_W, FRAME_H = 640, 480
BUILTIN = ("main", "simulation")
VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv")


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s[:40] or "camera"


def mask_source(source: str) -> str:
    """Hide a password in rtsp://user:pass@host/... before showing it."""
    return re.sub(r"(//[^:/@]+:)[^@/]+@", r"\1••••@", str(source))


class CameraStream:
    """Everything that belongs to one camera's video stream."""

    def __init__(self, cam_id: str, name: str, kind: str, base, zones_path: str):
        from analyzer import CrowdAnalyzer
        from evacuation_detector import EvacuationDetector
        from optimizer import AlertSystem
        from panic_detector import PanicDetector
        from tracker import PersonTracker
        from zones import ZoneManager

        self.id, self.name, self.kind = cam_id, name, kind
        self._base = base  # the shared SystemState (models)
        self.analyzer = CrowdAnalyzer(frame_shape=(FRAME_H, FRAME_W), history_length=60)
        use_ds = getattr(base.tracker_template, "use_deep_sort", False) if base.tracker_template else False
        self.tracker = PersonTracker(use_deep_sort=use_ds, max_disappeared=30)
        self.panic_detector = PanicDetector()
        self.evacuation_detector = EvacuationDetector()
        a = base.alert_template
        self.alert_system = AlertSystem(
            risk_threshold=a.risk_threshold, anomaly_threshold=a.anomaly_threshold,
            capacity_limit=a.capacity_limit, alert_ratio=a.alert_ratio,
            trend_window=a.trend_window, trend_min_slope=a.trend_min_slope)
        self.zones = ZoneManager(zones_path)
        self.lstm_predictor = self._own_lstm(base.lstm_predictor)
        self.count_corrector = self._own_corrector()

        self.stream_time = 0.0
        self.stream_source = None
        self.frame_count = 0
        self.last_update = 0.0
        self.history: deque = deque(maxlen=500)
        self.alert_log: deque = deque(maxlen=200)

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
        self.current_panic: Dict = {}
        self.current_zone_stats: List[Dict] = []
        self.current_count = 0
        self.last_annotated_jpeg: Optional[bytes] = None

    # The shared models stay shared; only their per stream memory is copied
    def _own_lstm(self, lstm):
        if lstm is None:
            return None
        own = copy.copy(lstm)
        own._history = deque(maxlen=lstm._history.maxlen)
        return own

    def _own_corrector(self):
        det = self._base.detector
        cc = getattr(det, "count_corrector", None) if det is not None else None
        if cc is None:
            return None
        own = copy.copy(cc)
        if hasattr(own, "reset"):
            own.reset()
        return own

    def refresh_models(self):
        """Call after the shared LSTM or count corrector changed."""
        self.lstm_predictor = self._own_lstm(self._base.lstm_predictor)
        self.count_corrector = self._own_corrector()

    def reset(self, source: str):
        """Forget the stream history (new video, or input switched)."""
        from tracker import PersonTracker
        self.analyzer.reset()
        self.tracker = PersonTracker(use_deep_sort=self.tracker.use_deep_sort, max_disappeared=30)
        self.panic_detector.reset()
        self.evacuation_detector.reset()
        self.stream_time = 0.0
        if self.lstm_predictor is not None:
            self.lstm_predictor._history.clear()
        if self.count_corrector is not None and hasattr(self.count_corrector, "reset"):
            self.count_corrector.reset()
        self.stream_source = source

    def summary(self) -> Dict:
        risk = self.current_risk or {}
        return {
            "id": self.id, "name": self.name, "kind": self.kind,
            "people_count": int(self.current_count),
            "risk_level": risk.get("risk_level", "low"),
            "alert_count": len(self.current_alerts),
            "critical": any(a.get("severity") == "critical" for a in self.current_alerts),
            "panic": bool(self.current_panic.get("is_panic")),
            "zones": len(self.zones.zones),
            "frames": self.frame_count,
            "last_update": self.last_update,
        }


class CameraReader:
    """Reads one server camera in a background thread and keeps the latest frame."""

    def __init__(self, cam_id: str, source: str):
        self.id = cam_id
        self.source = str(source).strip()
        self.status = "connecting"
        self.error = ""
        self.fps = 0.0
        self._frame: Optional[np.ndarray] = None
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"camera-{cam_id}")
        self._thread.start()

    @property
    def is_file(self) -> bool:
        return self.source.lower().endswith(VIDEO_EXT) and "://" not in self.source

    def _open(self):
        src = int(self.source) if self.source.isdigit() else self.source
        if isinstance(src, int) and os.name == "nt":
            cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)   # opens much faster on Windows
            if not cap.isOpened():
                cap = cv2.VideoCapture(src)
        else:
            cap = cv2.VideoCapture(src)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # live cameras: always the newest frame
        except Exception:
            pass
        return cap

    def _run(self):
        backoff = 1.0
        while not self._stop.is_set():
            if self.is_file and not os.path.exists(self.source):
                self.status, self.error = "offline", "file not found"
                self._stop.wait(5)
                continue
            cap = self._open()
            if not cap.isOpened():
                self.status, self.error = "offline", "could not open the camera"
                cap.release()
                self._stop.wait(backoff)
                backoff = min(backoff * 2, 30)
                continue
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_gap = 1.0 / file_fps if self.is_file and 1 <= file_fps <= 120 else 0
            self.status, self.error, backoff = "live", "", 1.0
            t_prev, failures = time.time(), 0
            while not self._stop.is_set():
                t0 = time.time()
                ok, frame = cap.read()
                if not ok or frame is None:
                    if self.is_file:   # loop the video
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                    if not ok or frame is None:
                        failures += 1
                        if failures > 25:
                            self.status, self.error = "offline", "stream stopped, reconnecting"
                            break
                        time.sleep(0.04)
                        continue
                failures = 0
                if frame.shape[:2] != (FRAME_H, FRAME_W):
                    frame = cv2.resize(frame, (FRAME_W, FRAME_H))
                with self._lock:
                    self._frame, self._seq = frame, self._seq + 1
                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now
                if frame_gap:   # play files at their real speed
                    self._stop.wait(max(0.0, frame_gap - (time.time() - t0)))
            cap.release()
            self._stop.wait(1)

    def latest(self):
        """(sequence number, frame) of the newest frame, or (seq, None)."""
        with self._lock:
            return self._seq, self._frame

    def stop(self):
        self._stop.set()


class CameraConfig:
    """Server cameras saved in data/cameras.json."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.cameras: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
            return [c for c in data if isinstance(c, dict) and c.get("id") and c.get("source") is not None]
        except (OSError, ValueError):
            return []

    def save(self):
        with self._lock:
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.cameras, f, indent=2)

    def get(self, cam_id: str) -> Optional[Dict]:
        return next((c for c in self.cameras if c["id"] == cam_id), None)

    def add(self, name: str, source: str, max_cameras: int = 16) -> Dict:
        name, source = (name or "").strip(), str(source or "").strip()
        if not name or len(name) > 60:
            raise ValueError("Give the camera a name (up to 60 characters)")
        if not source:
            raise ValueError("Enter the camera source: a webcam number, a stream URL or a video file path")
        if len(self.cameras) >= max_cameras:
            raise ValueError(f"At most {max_cameras} cameras")
        base, cam_id, i = slugify(name), slugify(name), 2
        taken = {c["id"] for c in self.cameras} | set(BUILTIN)
        while cam_id in taken:
            cam_id, i = f"{base}-{i}", i + 1
        cam = {"id": cam_id, "name": name, "source": source, "enabled": True}
        self.cameras.append(cam)
        self.save()
        return cam

    def update(self, cam_id: str, changes: Dict) -> Dict:
        cam = self.get(cam_id)
        if cam is None:
            raise KeyError(cam_id)
        if "name" in changes:
            name = str(changes["name"] or "").strip()
            if not name or len(name) > 60:
                raise ValueError("Give the camera a name (up to 60 characters)")
            cam["name"] = name
        if "source" in changes and str(changes["source"] or "").strip():
            cam["source"] = str(changes["source"]).strip()
        if "enabled" in changes:
            cam["enabled"] = bool(changes["enabled"])
        self.save()
        return cam

    def remove(self, cam_id: str):
        before = len(self.cameras)
        self.cameras = [c for c in self.cameras if c["id"] != cam_id]
        if len(self.cameras) == before:
            raise KeyError(cam_id)
        self.save()


def zones_path_for(data_dir: str, cam_id: str) -> str:
    """The browser camera keeps the original data/zones.json."""
    return os.path.join(data_dir, "zones.json" if cam_id == "main" else f"zones_{cam_id}.json")
