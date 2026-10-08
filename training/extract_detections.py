"""
Step 1: run YOLO once over every Mall frame and cache every box.

Boxes are kept down to conf 0.05 so the later training scripts can try any
confidence threshold without re-running YOLO.

Output: training/cache/detections.npz  (frame index, x1, y1, x2, y2, conf)
Usage:  python training/extract_detections.py
"""
import glob
import os
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(__file__), "..")
FRAMES = sorted(glob.glob(os.path.join(ROOT, "data/frames/frames/*.jpg")))
OUT = os.path.join(os.path.dirname(__file__), "cache", "detections.npz")


def main():
    if not FRAMES:
        sys.exit("No frames in data/frames/frames")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    model = YOLO(os.path.join(ROOT, "yolov8n.pt"))
    rows = []
    t0 = time.time()
    for i, path in enumerate(FRAMES):
        r = model(cv2.imread(path), conf=0.05, imgsz=1280, classes=[0], verbose=False)[0]
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
            rows.append((i, x1, y1, x2, y2, c))
        if (i + 1) % 100 == 0:
            el = time.time() - t0
            print(f"{i+1}/{len(FRAMES)} frames, {el:.0f}s elapsed, ~{el/(i+1)*(len(FRAMES)-i-1):.0f}s left", flush=True)
    np.savez_compressed(OUT, boxes=np.array(rows, dtype=np.float32), n_frames=len(FRAMES),
                        frame_names=np.array([os.path.basename(f) for f in FRAMES]))
    print(f"Saved {len(rows)} boxes to {OUT}")


if __name__ == "__main__":
    main()
