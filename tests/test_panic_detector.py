"""
Panic detector checks on the built-in simulator.
Run:  python -m pytest tests/      (or: python tests/test_panic_detector.py)
"""
import logging
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
logging.disable(logging.WARNING)

from panic_detector import PanicDetector  # noqa: E402
from simulator import CrowdSimulator  # noqa: E402
from tracker import PersonTracker  # noqa: E402


def run(scenario, seed, steps=1):
    np.random.seed(seed)
    sim = CrowdSimulator(640, 480, 15)
    sim.start_scenario(scenario)
    tracker = PersonTracker(use_deep_sort=False)
    det = PanicDetector()
    truth, flagged = [], []
    while sim.is_running:
        d = sim.step(steps=steps)
        r = det.update(tracker.update(d["detections"]), frames_elapsed=steps)
        truth.append(d["features"]["is_panic"])
        flagged.append(r["is_panic"])
    return np.array(truth), np.array(flagged)


def test_panic_event_is_detected_quickly():
    for seed in (100, 101):
        truth, flagged = run("panic_event", seed)
        assert flagged[truth].mean() > 0.85          # most panic frames caught
        start = int(np.argmax(truth))
        assert flagged[start:start + 15].any()       # within 1 second (15 frames)


def test_no_panic_in_calm_scenarios():
    for scenario in ("normal", "gradual_buildup", "sudden_surge"):
        _, flagged = run(scenario, 100)
        assert not flagged.any(), scenario


def test_no_false_alarm_outside_cooldown():
    truth, flagged = run("panic_event", 102)
    end = int(np.where(truth)[0][-1])
    cooldown = PanicDetector().clear_frames
    outside = flagged.copy()
    outside[truth] = False
    outside[end + 1:end + 1 + cooldown] = False
    assert not outside.any()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
