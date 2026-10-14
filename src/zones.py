"""
Per-zone crowd counting and alerts.

Zones are polygons drawn in the dashboard's Zone Editor, in frame
coordinates (640x480). Each has a type and a capacity:

  monitor     general area (food court, corridor); alerts at 80% / 100% of capacity
  entry       entrance; same as monitor
  exit        exit or emergency route; a crowded exit is reported as congested
  restricted  nobody should be here; anyone inside raises a critical alert

A person belongs to the zone their feet are in (bottom centre of their box),
which matches where they stand on the floor much better than the box centre.

YOLO misses some people, and the count corrector adds them back for the whole
frame. Zone counts are scaled by the same factor so that, across all zones,
they agree with the corrected total.

Zones are saved in data/zones.json.
"""
import json
import os
import threading
from typing import Dict, List, Optional

import numpy as np

TYPES = ("monitor", "entry", "exit", "restricted")
DEFAULT_CAPACITY = {"monitor": 15, "entry": 10, "exit": 8, "restricted": 0}
FRAME_W, FRAME_H = 640, 480
ALERT_RATIO = 0.8


def point_in_polygon(x: float, y: float, poly: np.ndarray) -> bool:
    """Ray casting; poly is an (N, 2) array."""
    inside = False
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi:
            inside = not inside
        j = i
    return inside


def foot_point(det: Dict):
    if 'bbox' in det:
        x1, y1, x2, y2 = det['bbox']
        return (x1 + x2) / 2.0, float(y2)
    cx, cy = det['center']
    return float(cx), float(cy)


class ZoneManager:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self.zones: List[Dict] = self._load()

    def _load(self) -> List[Dict]:
        try:
            with open(self.path, encoding="utf-8") as f:
                return self.validate(json.load(f))
        except (OSError, ValueError):
            return []

    @staticmethod
    def validate(zones) -> List[Dict]:
        if not isinstance(zones, list):
            raise ValueError("zones must be a list")
        if len(zones) > 50:
            raise ValueError("at most 50 zones")
        out, names = [], set()
        for i, z in enumerate(zones):
            if not isinstance(z, dict):
                raise ValueError(f"zone {i + 1} is not an object")
            ztype = z.get("type", "monitor")
            if ztype not in TYPES:
                raise ValueError(f"zone {i + 1}: type must be one of {TYPES}")
            pts = z.get("points") or []
            if len(pts) < 3 or len(pts) > 100:
                raise ValueError(f"zone {i + 1}: needs 3 to 100 points")
            try:
                pts = [[float(min(max(p[0], 0), FRAME_W)), float(min(max(p[1], 0), FRAME_H))] for p in pts]
            except (TypeError, IndexError, ValueError):
                raise ValueError(f"zone {i + 1}: points must be [x, y] pairs")
            name = str(z.get("name") or f"Zone {i + 1}").strip()[:40] or f"Zone {i + 1}"
            if name.lower() in names:
                name = f"{name} ({i + 1})"
            names.add(name.lower())
            cap = z.get("capacity")
            cap = DEFAULT_CAPACITY[ztype] if cap in (None, "") else int(cap)
            if cap < 0 or cap > 10000:
                raise ValueError(f"zone {i + 1}: capacity must be 0 to 10000")
            out.append({"id": str(z.get("id") or f"z{i + 1}")[:40], "name": name,
                        "type": ztype, "points": pts, "capacity": cap})
        return out

    def set_zones(self, zones) -> List[Dict]:
        clean = self.validate(zones)
        with self._lock:
            self.zones = clean
            os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(clean, f, indent=2)
            os.replace(tmp, self.path)
        return clean

    def analyze(self, detections: List[Dict], total_count: Optional[int] = None) -> List[Dict]:
        """Live stats for every saved zone."""
        zones = self.zones
        if not zones:
            return []
        feet = [foot_point(d) for d in detections]
        scale = 1.0
        if total_count is not None and len(detections) > 0 and total_count > len(detections):
            scale = total_count / len(detections)
        out = []
        for z in zones:
            poly = np.asarray(z["points"], dtype=float)
            raw = sum(point_in_polygon(x, y, poly) for x, y in feet)
            count = int(round(raw * scale)) if raw else 0
            cap = z["capacity"]
            if z["type"] == "restricted":
                risk = "high" if count > 0 else "low"
                util = None
            else:
                util = count / cap if cap > 0 else None
                risk = ("high" if util is not None and util > 1 else
                        "medium" if util is not None and util >= ALERT_RATIO else "low")
            out.append({**z, "people_count": count, "detected": raw,
                        "utilization_percent": round(util * 100, 1) if util is not None else None,
                        "risk": risk, "label": z["name"]})
        return out


def zone_alerts(stats: List[Dict], timestamp: float) -> List[Dict]:
    """Alerts for zones over capacity, congested exits and restricted areas."""
    alerts = []
    for z in stats:
        n, cap, name = z["people_count"], z["capacity"], z["name"]
        data = {"zone": name, "zone_type": z["type"], "count": n, "capacity": cap}
        if z["type"] == "restricted":
            if n > 0:
                alerts.append({"type": "zone_restricted", "severity": "critical", "timestamp": timestamp,
                               "message": f"{n} {'person' if n == 1 else 'people'} in restricted zone '{name}'.",
                               "data": data})
            continue
        if cap <= 0:
            continue
        if n > cap:
            msg = (f"Exit '{name}' is congested: {n}/{cap} people. Keep it clear."
                   if z["type"] == "exit" else f"Zone '{name}' is over capacity: {n}/{cap} people.")
            alerts.append({"type": "zone_capacity_exceeded", "severity": "critical",
                           "timestamp": timestamp, "message": msg, "data": data})
        elif n >= cap * ALERT_RATIO:
            alerts.append({"type": "zone_capacity_threshold", "severity": "warning", "timestamp": timestamp,
                           "message": f"Zone '{name}' is filling up: {n}/{cap} people ({n / cap:.0%}).",
                           "data": data})
    return alerts
