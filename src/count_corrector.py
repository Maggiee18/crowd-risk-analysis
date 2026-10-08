"""
Count corrector: maps raw YOLO detections to a calibrated people count.

YOLOv8n undercounts the Mall dataset (small, occluded, top-down people).
This model learns from mall_gt.mat how the raw boxes relate to the true
count, using features like the number of boxes at several confidence
levels and where in the frame (near/far) they are.

Trained by training/train_models.py, saved to models/count_corrector.joblib.
"""
from typing import List, Dict, Optional
import os

import joblib
import numpy as np

CONF_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
FEATURE_NAMES = (
    [f"n_conf_{c}" for c in CONF_LEVELS]
    + ["conf_sum", "n_top", "n_mid", "n_bottom",
       "mean_area_norm", "median_h_norm", "frac_small"]
)


def box_features(boxes: np.ndarray, frame_h: int = 480, frame_w: int = 640) -> np.ndarray:
    """
    boxes: array (N, 5) of x1, y1, x2, y2, conf (any confidence >= 0.05)
    returns: 1D feature vector matching FEATURE_NAMES
    """
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 5)
    conf = boxes[:, 4]
    feats = [float(np.sum(conf >= c)) for c in CONF_LEVELS]
    feats.append(float(conf.sum()))

    keep = boxes[conf >= 0.15]
    if len(keep):
        cy = (keep[:, 1] + keep[:, 3]) / 2 / frame_h
        h = (keep[:, 3] - keep[:, 1]) / frame_h
        area = (keep[:, 2] - keep[:, 0]) * (keep[:, 3] - keep[:, 1]) / (frame_h * frame_w)
        feats += [float(np.sum(cy < 1 / 3)), float(np.sum((cy >= 1 / 3) & (cy < 2 / 3))),
                  float(np.sum(cy >= 2 / 3)), float(area.mean()), float(np.median(h)),
                  float(np.mean(h < 0.12))]
    else:
        feats += [0.0] * 6
    return np.array(feats, dtype=np.float32)


def detections_to_boxes(detections: List[Dict]) -> np.ndarray:
    return np.array([[*d["bbox"], d["confidence"]] for d in detections], dtype=np.float32).reshape(-1, 5)


class CountCorrector:
    def __init__(self, model=None):
        self.model = model

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, boxes: np.ndarray, frame_h: int = 480, frame_w: int = 640) -> int:
        x = box_features(boxes, frame_h, frame_w).reshape(1, -1)
        return max(0, int(round(float(self.model.predict(x)[0]))))

    def save(self, path: str):
        joblib.dump({"model": self.model, "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: str) -> Optional["CountCorrector"]:
        if not os.path.exists(path):
            return None
        return cls(joblib.load(path)["model"])
