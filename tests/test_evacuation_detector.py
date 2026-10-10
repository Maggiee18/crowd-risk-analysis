"""
Evacuation detector checks on the simulator and the real Mall counts.
Run:  python -m pytest tests/      (or: python tests/test_evacuation_detector.py)
"""
import logging
import os
import sys

import numpy as np
import scipy.io as sio

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
logging.disable(logging.WARNING)

from evacuation_detector import EvacuationDetector  # noqa: E402
from simulator import CrowdSimulator  # noqa: E402


def first_alert(scenario, seed, steps=1):
    np.random.seed(seed)
    sim = CrowdSimulator(640, 480, 15)
    sim.start_scenario(scenario)
    det, t = EvacuationDetector(), 0.0
    while sim.is_running:
        d = sim.step(steps=steps)
        t += steps / 15
        if det.update(d["people_count"], t)["is_evacuation"]:
            return t
    return None


def test_evacuation_is_detected():
    for seed in (200, 201, 202):
        t = first_alert("evacuation", seed)
        assert t is not None and t < 20, (seed, t)   # scenario lasts 25 s


def test_no_evacuation_in_other_scenarios():
    for scenario in ("normal", "gradual_buildup", "sudden_surge", "panic_event", "concert"):
        assert first_alert(scenario, 200) is None, scenario


def test_no_false_alarm_on_real_mall_counts():
    counts = sio.loadmat(os.path.join(ROOT, "data", "mall_gt.mat"))["count"].reshape(-1)
    for fps in (2.0, 1.0):
        det = EvacuationDetector()
        assert not any(det.update(c, i / fps)["is_evacuation"] for i, c in enumerate(counts)), fps


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
