"""
Multi-Object Tracking Module
Implements Deep SORT tracking for persistent person identification across frames.
Falls back to centroid-based tracking if deep_sort_realtime is unavailable.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
import logging
import time

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lightweight centroid tracker (always available, no extra deps)
# ---------------------------------------------------------------------------

class CentroidTracker:
    """Simple centroid-based tracker using distance matching."""

    def __init__(self, max_disappeared: int = 30, max_distance: float = 80.0):
        self.next_id = 0
        self.objects: Dict[int, np.ndarray] = {}
        self.disappeared: Dict[int, int] = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def _register(self, centroid: np.ndarray) -> int:
        obj_id = self.next_id
        self.objects[obj_id] = centroid
        self.disappeared[obj_id] = 0
        self.next_id += 1
        return obj_id

    def _deregister(self, obj_id: int):
        del self.objects[obj_id]
        del self.disappeared[obj_id]

    def update(self, detections: List[Dict]) -> Dict[int, Dict]:
        """
        Match new detections to existing tracked objects.

        Args:
            detections: List of detection dicts with 'bbox' [x1,y1,x2,y2]

        Returns:
            Dict mapping track_id -> {'bbox': [...], 'center': [...]}
        """
        if len(detections) == 0:
            for obj_id in list(self.disappeared.keys()):
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)
            return {}

        # Compute centroids for incoming detections
        input_centroids = np.array([
            [(d['bbox'][0] + d['bbox'][2]) / 2,
             (d['bbox'][1] + d['bbox'][3]) / 2]
            for d in detections
        ])

        if len(self.objects) == 0:
            result = {}
            for i, centroid in enumerate(input_centroids):
                tid = self._register(centroid)
                result[tid] = {
                    'bbox': detections[i]['bbox'],
                    'center': centroid.tolist(),
                    'confidence': detections[i].get('confidence', 0.0)
                }
            return result

        # Distance matrix between existing objects and new detections
        obj_ids = list(self.objects.keys())
        obj_centroids = np.array(list(self.objects.values()))

        dists = np.linalg.norm(
            obj_centroids[:, np.newaxis] - input_centroids[np.newaxis, :], axis=2
        )

        # Greedy assignment (Hungarian would be better but this is lightweight)
        rows = dists.min(axis=1).argsort()
        cols = dists.argmin(axis=1)

        used_rows = set()
        used_cols = set()
        result = {}

        for row in rows:
            col = cols[row]
            if row in used_rows or col in used_cols:
                continue
            if dists[row, col] > self.max_distance:
                continue

            obj_id = obj_ids[row]
            self.objects[obj_id] = input_centroids[col]
            self.disappeared[obj_id] = 0
            result[obj_id] = {
                'bbox': detections[col]['bbox'],
                'center': input_centroids[col].tolist(),
                'confidence': detections[col].get('confidence', 0.0)
            }
            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects
        for row in range(len(obj_ids)):
            if row not in used_rows:
                obj_id = obj_ids[row]
                self.disappeared[obj_id] += 1
                if self.disappeared[obj_id] > self.max_disappeared:
                    self._deregister(obj_id)

        # Register new detections that were not matched
        for col in range(len(input_centroids)):
            if col not in used_cols:
                tid = self._register(input_centroids[col])
                result[tid] = {
                    'bbox': detections[col]['bbox'],
                    'center': input_centroids[col].tolist(),
                    'confidence': detections[col].get('confidence', 0.0)
                }

        return result


# ---------------------------------------------------------------------------
# Deep SORT wrapper (optional, requires deep_sort_realtime + torch)
# ---------------------------------------------------------------------------

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort as _DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    logger.info("deep_sort_realtime not installed — using centroid tracker")


class DeepSORTWrapper:
    """Wrapper around deep_sort_realtime for consistent API."""

    def __init__(self, max_age: int = 30, n_init: int = 3,
                 max_cosine_distance: float = 0.3):
        if not DEEPSORT_AVAILABLE:
            raise ImportError("deep_sort_realtime is not installed")
        self.tracker = _DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance
        )

    def update(self, detections: List[Dict], frame=None) -> Dict[int, Dict]:
        """
        Update tracker with new detections.

        Args:
            detections: List of dicts with 'bbox' [x1,y1,x2,y2] and 'confidence'
            frame: Current video frame (needed for appearance features)

        Returns:
            Dict mapping track_id -> {'bbox': [...], 'center': [...]}
        """
        if len(detections) == 0:
            self.tracker.update_tracks([], frame=frame)
            return {}

        # deep_sort_realtime expects [[x1,y1,w,h], ...] + confidences
        bbs = []
        for d in detections:
            x1, y1, x2, y2 = d['bbox']
            bbs.append(([x1, y1, x2 - x1, y2 - y1], d.get('confidence', 0.8), 'person'))

        tracks = self.tracker.update_tracks(bbs, frame=frame)

        result = {}
        for track in tracks:
            if not track.is_confirmed():
                continue
            tid = track.track_id
            ltrb = track.to_ltrb()  # [left, top, right, bottom]
            bbox = [int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])]
            center = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            result[int(tid)] = {
                'bbox': bbox,
                'center': center,
                'confidence': track.det_conf if hasattr(track, 'det_conf') else 0.0
            }

        return result


# ---------------------------------------------------------------------------
# Main PersonTracker that unifies both approaches + velocity/trajectory
# ---------------------------------------------------------------------------

class PersonTracker:
    """
    High-level person tracker that manages tracking, velocity estimation,
    and trajectory history for all tracked individuals.
    """

    def __init__(self, use_deep_sort: bool = True,
                 max_disappeared: int = 30,
                 trajectory_length: int = 60):
        """
        Args:
            use_deep_sort: Try to use Deep SORT (falls back to centroid if unavailable)
            max_disappeared: Frames before a track is dropped
            trajectory_length: Max number of past positions to store per track
        """
        self.use_deep_sort = use_deep_sort and DEEPSORT_AVAILABLE

        if self.use_deep_sort:
            try:
                self.tracker = DeepSORTWrapper(max_age=max_disappeared)
                logger.info("Using Deep SORT tracker")
            except Exception as e:
                # e.g. deep_sort_realtime needs pkg_resources, which newer
                # setuptools no longer ships. Don't let that kill the server.
                logger.warning(f"Deep SORT unavailable ({e}), using centroid tracker")
                self.use_deep_sort = False
        if not self.use_deep_sort:
            self.tracker = CentroidTracker(max_disappeared=max_disappeared)
            logger.info("Using Centroid tracker")

        self.trajectory_length = trajectory_length
        # track_id -> deque of (timestamp, center_x, center_y)
        self.trajectories: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=trajectory_length)
        )
        # track_id -> (vx, vy) in pixels/second
        self.velocities: Dict[int, Tuple[float, float]] = {}
        self._last_update_time = time.time()

    def update(self, detections: List[Dict], frame=None) -> Dict[int, Dict]:
        """
        Update tracker with new detections and compute velocities.

        Args:
            detections: List of dicts with 'bbox', 'confidence', 'center'
            frame: Current BGR frame (needed for Deep SORT appearance model)

        Returns:
            Dict track_id -> {
                'bbox': [x1,y1,x2,y2],
                'center': [cx, cy],
                'velocity': [vx, vy],   # px/sec
                'speed': float,          # px/sec
                'trajectory': [[cx,cy], ...],
                'confidence': float
            }
        """
        now = time.time()
        dt = now - self._last_update_time
        self._last_update_time = now

        # Run underlying tracker
        if self.use_deep_sort:
            tracks = self.tracker.update(detections, frame=frame)
        else:
            tracks = self.tracker.update(detections)

        result = {}
        active_ids = set()

        for tid, info in tracks.items():
            active_ids.add(tid)
            cx, cy = info['center']
            self.trajectories[tid].append((now, cx, cy))

            # Compute velocity from recent trajectory
            traj = self.trajectories[tid]
            vx, vy = 0.0, 0.0
            if len(traj) >= 2 and dt > 0:
                _, px, py = traj[-2]
                vx = (cx - px) / dt
                vy = (cy - py) / dt
            self.velocities[tid] = (vx, vy)

            speed = np.sqrt(vx ** 2 + vy ** 2)

            result[tid] = {
                'bbox': info['bbox'],
                'center': [cx, cy],
                'velocity': [round(vx, 2), round(vy, 2)],
                'speed': round(speed, 2),
                'trajectory': [[p[1], p[2]] for p in traj],
                'confidence': info.get('confidence', 0.0)
            }

        # Clean up trajectories for disappeared tracks
        disappeared = set(self.trajectories.keys()) - active_ids
        for tid in disappeared:
            if tid in self.trajectories and len(self.trajectories[tid]) > 0:
                last_time = self.trajectories[tid][-1][0]
                if now - last_time > 10:  # 10 seconds grace period
                    del self.trajectories[tid]
                    self.velocities.pop(tid, None)

        return result

    def get_average_speed(self) -> float:
        """Get average speed of all currently tracked people."""
        speeds = [np.sqrt(v[0]**2 + v[1]**2) for v in self.velocities.values()]
        return float(np.mean(speeds)) if speeds else 0.0

    def get_direction_histogram(self, n_bins: int = 8) -> Dict[str, int]:
        """
        Compute histogram of movement directions for all tracked people.
        Useful for detecting direction conflicts (counter-flow).
        """
        directions = {
            'N': 0, 'NE': 0, 'E': 0, 'SE': 0,
            'S': 0, 'SW': 0, 'W': 0, 'NW': 0
        }
        dir_labels = list(directions.keys())

        for vx, vy in self.velocities.values():
            speed = np.sqrt(vx**2 + vy**2)
            if speed < 5.0:  # Ignore nearly stationary
                continue
            angle = np.arctan2(-vy, vx)  # -vy because y-axis is inverted
            angle_deg = np.degrees(angle) % 360
            idx = int((angle_deg + 22.5) / 45) % 8
            directions[dir_labels[idx]] += 1

        return directions

    def detect_direction_conflicts(self, threshold: float = 0.3) -> bool:
        """
        Detect if there are significant counter-flow groups
        (people moving in opposing directions).

        Args:
            threshold: Minimum ratio of opposing flow to trigger conflict

        Returns:
            True if direction conflict detected
        """
        hist = self.get_direction_histogram()
        total = sum(hist.values())
        if total < 4:
            return False

        # Check opposing pairs
        opposites = [('N', 'S'), ('NE', 'SW'), ('E', 'W'), ('NW', 'SE')]
        for d1, d2 in opposites:
            c1 = hist[d1] / total
            c2 = hist[d2] / total
            if c1 > threshold and c2 > threshold:
                return True
        return False

    def get_tracking_summary(self) -> Dict:
        """Get summary statistics of current tracking state."""
        return {
            'active_tracks': len(self.velocities),
            'average_speed': self.get_average_speed(),
            'direction_histogram': self.get_direction_histogram(),
            'direction_conflict': self.detect_direction_conflicts(),
            'total_tracks_seen': (
                self.tracker.tracker.next_id if hasattr(self.tracker, 'tracker') 
                and hasattr(self.tracker.tracker, 'next_id')
                else self.tracker.next_id if hasattr(self.tracker, 'next_id')
                else 0
            )
        }


if __name__ == "__main__":
    # Quick test with synthetic detections
    tracker = PersonTracker(use_deep_sort=False)

    for frame_idx in range(10):
        # Simulate 3 people moving
        dets = [
            {'bbox': [100 + frame_idx*5, 100, 140 + frame_idx*5, 200],
             'confidence': 0.9, 'center': [120 + frame_idx*5, 150]},
            {'bbox': [300, 200 + frame_idx*3, 340, 300 + frame_idx*3],
             'confidence': 0.85, 'center': [320, 250 + frame_idx*3]},
            {'bbox': [500 - frame_idx*4, 150, 540 - frame_idx*4, 250],
             'confidence': 0.88, 'center': [520 - frame_idx*4, 200]},
        ]
        tracks = tracker.update(dets)
        time.sleep(0.05)

    summary = tracker.get_tracking_summary()
    print(f"Active tracks: {summary['active_tracks']}")
    print(f"Avg speed: {summary['average_speed']:.1f} px/s")
    print(f"Direction conflict: {summary['direction_conflict']}")
    print(f"Directions: {summary['direction_histogram']}")
