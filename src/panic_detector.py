"""
Panic detection from crowd movement.

Panic shows up as a sudden change in how people move, not in how many there
are: most people speed up at once and move in scattered directions. This
detector compares the crowd's current movement against its own recent
"normal", so it adapts to each camera and needs no training data.

Speeds are measured in body heights per video frame (displacement divided by
the person's box height). That makes them independent of camera resolution,
perspective (far people are small) and how fast the computer processes frames.

Signals, per frame:
  fast_fraction share of people moving faster than `fast_factor` x the fast
                end of normal (90th percentile of recent calm speeds). A
                median is not used: in a mall many people stand while others
                walk, and the median jumps whenever that mix changes.
  speed_ratio   75th percentile speed now / 75th percentile in calm frames
  coherence     0..1, how aligned movement directions are (1 = all the same way)

Robustness: a tracker that mixes two people up (an ID swap) produces one
huge jump, then normal motion again. Someone really panicking stays fast.
So a person's speed is the smaller of their last two frame-to-frame speeds,
and single jumps over `max_jump` x the normal fast end are ignored.

Panic is raised when, for `confirm_frames` frames in a row,
  fast_fraction >= fast_fraction_threshold AND speed_ratio >= ratio_threshold
and cleared after `clear_frames` calm frames (hysteresis, so it doesn't flicker).
Low coherence (scattering) makes the alert critical; an aligned rush (everyone
running the same way) is still reported, as a 'rush'.
"""
from collections import deque
from typing import Dict

import numpy as np


class PanicDetector:
    def __init__(self,
                 ratio_threshold: float = 2.0,
                 fast_fraction_threshold: float = 0.4,
                 fast_factor: float = 1.5,
                 confirm_frames: int = 3,
                 clear_frames: int = 8,
                 baseline_size: int = 3000,
                 min_tracks: int = 4,
                 min_baseline_frames: int = 15,
                 speed_floor: float = 0.005,
                 max_jump: float = 5.0):
        self.ratio_threshold = ratio_threshold
        self.fast_factor = fast_factor
        self.fast_fraction_threshold = fast_fraction_threshold
        self.confirm_frames = confirm_frames
        self.clear_frames = clear_frames
        self.min_tracks = min_tracks
        self.min_baseline_frames = min_baseline_frames
        self.speed_floor = speed_floor  # body heights/frame, ignores jitter
        self.max_jump = max_jump
        self.baseline = deque(maxlen=baseline_size)  # individual speeds, calm frames
        self._baseline_frames = 0
        self.reset()

    def reset(self):
        """Forget everything (call when switching video or scenario)."""
        self.baseline.clear()
        self._baseline_frames = 0
        self._last: Dict[int, tuple] = {}   # track id -> (frame_idx, cx, cy, h, speed)
        self._frame = 0
        self._hot = 0
        self._calm = 0
        self.active = False

    def update(self, tracks: Dict, frames_elapsed: float = 1.0) -> Dict:
        """
        tracks: {track_id: {'bbox': [x1,y1,x2,y2], 'center': [cx, cy], ...}}
                (the output of PersonTracker.update)
        frames_elapsed: video frames since the previous call (a simulation
                that skips ahead passes >1; normal video passes 1)
        """
        self._frame += 1
        frames_elapsed = max(float(frames_elapsed), 1e-6)

        base_fast = float(np.percentile(self.baseline, 90)) if len(self.baseline) >= 20 else None
        speeds, dirs = [], []
        seen = {}
        for tid, t in tracks.items():
            x1, y1, x2, y2 = t['bbox']
            h = max(float(y2 - y1), 1.0)
            cx, cy = float(t['center'][0]), float(t['center'][1])
            prev = self._last.get(tid)
            if prev is None or prev[0] != self._frame - 1:
                seen[tid] = (self._frame, cx, cy, h, None)
                continue  # need the same person in the previous frame
            dx, dy = cx - prev[1], cy - prev[2]
            s = np.hypot(dx, dy) / ((h + prev[3]) / 2) / frames_elapsed
            if base_fast is not None and s > self.max_jump * max(base_fast, self.speed_floor):
                seen[tid] = (self._frame, cx, cy, h, None)  # ID swap, not motion
                continue
            seen[tid] = (self._frame, cx, cy, h, s)
            if prev[4] is None:
                continue  # need two speeds in a row
            speeds.append(min(s, prev[4]))
            dirs.append(np.arctan2(dy, dx))
        self._last = seen

        result = {
            'is_panic': self.active, 'kind': None, 'speed_ratio': 0.0,
            'fast_fraction': 0.0, 'coherence': 0.0,
            'median_speed': 0.0, 'baseline_speed': 0.0, 'tracked': len(speeds),
        }
        if len(speeds) < self.min_tracks:
            return result

        speeds = np.array(speeds)
        result['median_speed'] = round(float(np.median(speeds)), 4)

        if self._baseline_frames < self.min_baseline_frames:
            # still learning what "normal" looks like for this camera
            self.baseline.extend(speeds.tolist())
            self._baseline_frames += 1
            return result

        pool = np.asarray(self.baseline)
        fast_end = max(float(np.percentile(pool, 90)), self.speed_floor)
        typical = max(float(np.percentile(pool, 75)), self.speed_floor)
        ratio = float(np.percentile(speeds, 75)) / typical
        fast = float(np.mean(speeds > self.fast_factor * fast_end))
        moving = speeds > self.speed_floor
        if moving.sum() >= 2:
            d = np.array(dirs)[moving]
            coherence = float(np.hypot(np.cos(d).mean(), np.sin(d).mean()))
        else:
            coherence = 1.0

        hot = fast >= self.fast_fraction_threshold and ratio >= self.ratio_threshold
        if hot:
            self._hot += 1
            self._calm = 0
        else:
            self._calm += 1
            self._hot = 0
        if not self.active and self._hot >= self.confirm_frames:
            self.active = True
        elif self.active and self._calm >= self.clear_frames:
            self.active = False

        # Only learn "normal" from calm frames, so a long panic doesn't
        # become the new baseline.
        if not hot and not self.active:
            self.baseline.extend(speeds.tolist())
            self._baseline_frames += 1

        result.update({
            'is_panic': self.active,
            'kind': (('scatter' if coherence < 0.5 else 'rush') if self.active else None),
            'speed_ratio': round(ratio, 2),
            'fast_fraction': round(fast, 2),
            'coherence': round(coherence, 2),
            'baseline_speed': round(fast_end, 4),
        })
        return result
