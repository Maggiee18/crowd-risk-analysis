"""
Evacuation detection from the people count.

An evacuation shows up as a large crowd emptying out fast and steadily. The
movement itself need not look like panic, so panic_detector.py can miss it.

Real venues also swing a lot as groups walk in and out of view: in the Mall
video the count drops by more than 50% within 10 seconds dozens of times.
So all of these must hold for `confirm_s` seconds:

  1. the crowd had at least `min_peak` people in the last `window_s` seconds
  2. it has since lost at least `min_drop` of them (default 60%)
  3. the decline is steady, not a dip: rank correlation of count vs time
     over the window <= `max_trend` (default -0.7)

Time is in seconds (simulation time, or wall clock for a camera), so the
detector works at any frame rate.

Tuned on the built-in simulator and the Mall dataset: Evacuation is flagged
9 to 11 s into its 25 s run, with zero alerts in every other scenario and on
all 2000 Mall frames (both our counts and the ground truth counts).
"""
from collections import deque
from typing import Dict, Optional

import numpy as np


def _rank_corr(y: np.ndarray) -> float:
    """Spearman correlation of y against time (its index)."""
    if len(y) < 3 or np.ptp(y) == 0:
        return 0.0
    ry = np.argsort(np.argsort(y, kind="stable"), kind="stable").astype(float)
    # average ranks for ties keep it close to scipy's spearmanr
    _, inv, counts = np.unique(y, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ry)
    ry = (sums / counts)[inv]
    rx = np.arange(len(y), dtype=float)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


class EvacuationDetector:
    def __init__(self, window_s: float = 10.0, min_drop: float = 0.6,
                 max_trend: float = -0.7, min_peak: int = 15,
                 confirm_s: float = 2.0, clear_s: float = 5.0,
                 smooth_s: float = 1.0):
        self.window_s = window_s
        self.min_drop = min_drop
        self.max_trend = max_trend
        self.min_peak = min_peak
        self.confirm_s = confirm_s
        self.clear_s = clear_s
        self.smooth_s = smooth_s
        self.reset()

    def reset(self):
        self._raw = deque()       # (t, count)
        self._smooth = deque()    # (t, smoothed count)
        self._hot_since: Optional[float] = None
        self._calm_since: Optional[float] = None
        self.active = False

    def update(self, count: float, t: float) -> Dict:
        """count: people in this frame; t: time in seconds (monotonic)."""
        self._raw.append((t, float(count)))
        while self._raw and self._raw[0][0] < t - self.smooth_s:
            self._raw.popleft()
        now = float(np.median([c for _, c in self._raw]))
        self._smooth.append((t, now))
        while self._smooth and self._smooth[0][0] < t - self.window_s:
            self._smooth.popleft()

        seg = np.array([c for _, c in self._smooth])
        i_peak = int(np.argmax(seg))
        peak = float(seg[i_peak])
        drop = (peak - now) / peak if peak > 0 else 0.0
        trend = _rank_corr(seg)
        span = t - self._smooth[0][0]

        hot = (peak >= self.min_peak and drop >= self.min_drop and trend <= self.max_trend)
        if hot:
            self._calm_since = None
            if self._hot_since is None:
                self._hot_since = t
            if not self.active and t - self._hot_since >= self.confirm_s:
                self.active = True
        else:
            self._hot_since = None
            if self.active:
                if self._calm_since is None:
                    self._calm_since = t
                if t - self._calm_since >= self.clear_s:
                    self.active = False

        return {
            'is_evacuation': self.active,
            'peak_count': int(round(peak)),
            'current_count': int(round(now)),
            'drop': round(drop, 2),
            'trend': round(trend, 2),
            'seconds_since_peak': round(t - self._smooth[i_peak][0], 1),
            'window_seconds': round(span, 1),
        }
