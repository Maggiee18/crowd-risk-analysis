"""Tests for camera settings and the camera reader (src/cameras.py). Run:  python tests/test_cameras.py"""
import os
import sys
import time

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cameras import CameraConfig, CameraReader, mask_source, zones_path_for  # noqa: E402


def test_add_rename_disable_remove(tmp_path):
    cfg = CameraConfig(str(tmp_path / "cameras.json"))
    a = cfg.add("Main Gate", "0")
    b = cfg.add("Main Gate", "rtsp://u:p@10.0.0.2/s")
    assert a["id"] == "main-gate" and b["id"] == "main-gate-2"
    cfg.update("main-gate", {"name": "North gate", "enabled": False})
    again = CameraConfig(str(tmp_path / "cameras.json"))   # saved to disk
    assert again.get("main-gate")["name"] == "North gate"
    assert again.get("main-gate")["enabled"] is False
    again.remove("main-gate-2")
    assert [c["id"] for c in again.cameras] == ["main-gate"]
    with pytest.raises(KeyError):
        again.remove("nope")


def test_builtin_ids_and_bad_input(tmp_path):
    cfg = CameraConfig(str(tmp_path / "cameras.json"))
    assert cfg.add("Main", "1")["id"] == "main-2"          # 'main' is the browser camera
    assert cfg.add("Simulation", "2")["id"] == "simulation-2"
    with pytest.raises(ValueError):
        cfg.add("", "0")
    with pytest.raises(ValueError):
        cfg.add("Gate", "  ")


def test_password_is_masked(tmp_path=None):
    assert mask_source("rtsp://admin:hunter2@192.168.1.20:554/s1") == "rtsp://admin:••••@192.168.1.20:554/s1"
    assert mask_source("0") == "0"


def test_zone_files_per_camera(tmp_path):
    assert zones_path_for(str(tmp_path), "main").endswith("zones.json")
    assert zones_path_for(str(tmp_path), "gate").endswith("zones_gate.json")


def test_reader_plays_and_loops_a_video_file(tmp_path):
    path = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 30, (320, 240))
    for i in range(10):
        out.write(np.full((240, 320, 3), i * 20, np.uint8))
    out.release()
    reader = CameraReader("t", path)
    try:
        deadline = time.time() + 5
        while time.time() < deadline and reader.latest()[0] < 15:   # more frames than the file has
            time.sleep(0.05)
        seq, frame = reader.latest()
        assert reader.status == "live"
        assert seq >= 15 and frame.shape == (480, 640, 3)
    finally:
        reader.stop()


def test_reader_reports_a_missing_camera(tmp_path):
    reader = CameraReader("t", str(tmp_path / "missing.mp4"))
    try:
        time.sleep(0.3)
        assert reader.status == "offline" and reader.error
    finally:
        reader.stop()


if __name__ == "__main__":
    import pathlib
    import tempfile
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            with tempfile.TemporaryDirectory() as d:
                fn(pathlib.Path(d))
            print("ok", name)
    print("All camera tests passed")
