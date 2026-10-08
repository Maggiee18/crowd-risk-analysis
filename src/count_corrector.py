"""
Count corrector: turns YOLO output into a calibrated people count.

YOLOv8n undercounts the Mall dataset (small, occluded, top-down people), and
counting boxes alone tops out at ~2.4 people error per frame. The corrector
combines two kinds of features:

1. Box features: number of YOLO boxes at several confidence levels and where
   in the frame they are.
2. Image features (v2): YOLO's own backbone feature maps (stride 8/16/32),
   summed over the frame and weighted by the dataset's perspective map, then
   compressed with PCA. These let the model "see" people YOLO missed.

A ridge regression on both gives the count, followed by light temporal
smoothing (EMA) because a video's count changes slowly.

The model is stored as plain numpy arrays (models/count_corrector.npz), so it
loads identically with any scikit-learn version.

Trained by training/train_models.py.
"""
import os
from typing import Dict, List, Optional

import cv2
import numpy as np

CONF_LEVELS = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5)
FEATURE_NAMES = (
    [f"n_conf_{c}" for c in CONF_LEVELS]
    + ["conf_sum", "n_top", "n_mid", "n_bottom",
       "mean_area_norm", "median_h_norm", "frac_small"]
)

EMB_H, EMB_W = 30, 40      # stride-16 grid of a 480x640 frame
EMB_SIZE = (640, 480)      # frames are resized to this before the backbone pass


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


def load_perspective(path: str) -> np.ndarray:
    """Perspective map resized to the stride-16 grid and normalised to mean 1."""
    try:
        import scipy.io as sio
        pmap = sio.loadmat(path)["pMapN"].astype(np.float32)
        p = cv2.resize(pmap, (EMB_W, EMB_H), interpolation=cv2.INTER_AREA)
        return (p / p.mean()).reshape(-1, 1)
    except Exception:
        return np.ones((EMB_H * EMB_W, 1), dtype=np.float32)


def frame_embedding(yolo, frame: np.ndarray, persp: np.ndarray) -> np.ndarray:
    """
    Run YOLO's backbone (layers 0-9) on the frame and return the per-frame
    image feature vector: per-cell features [f8, f16, f32, f*persp, 1, persp]
    averaged over the 30x40 grid. `yolo` is an ultralytics.YOLO object.
    """
    import torch
    import torch.nn.functional as F

    img = cv2.resize(frame, EMB_SIZE) if frame.shape[:2] != (EMB_SIZE[1], EMB_SIZE[0]) else frame
    x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().unsqueeze(0) / 255
    layers = yolo.model.model
    outs = {}
    with torch.no_grad():
        for i in range(10):
            x = layers[i](x)
            outs[i] = x
        f8 = F.avg_pool2d(outs[4], 2)[0]
        f16 = outs[6][0]
        f32 = F.interpolate(outs[9], size=(EMB_H, EMB_W), mode="bilinear", align_corners=False)[0]
        cells = torch.cat([f8, f16, f32], 0).reshape(-1, EMB_H * EMB_W).T.numpy()
    cells = np.concatenate([cells, cells * persp, np.ones((len(cells), 1), np.float32), persp], 1)
    return cells.mean(0).astype(np.float32)


class CountCorrector:
    """
    v2 (image features): model is a dict of numpy arrays with keys
      pca_mean, pca_components, x_mean, x_scale, coef, intercept, ema_alpha
    v1 (box features only, older joblib files): model is a sklearn regressor.
    """

    def __init__(self, model=None, params: Optional[Dict[str, np.ndarray]] = None):
        self.model = model
        self.params = params
        self._ema = None

    @property
    def is_trained(self) -> bool:
        return self.model is not None or self.params is not None

    @property
    def needs_embedding(self) -> bool:
        return self.params is not None

    def reset(self):
        """Forget the smoothing state (call when switching video/stream)."""
        self._ema = None

    # ---- v2 helpers (used by training too) ----
    @staticmethod
    def build_x(box_feats: np.ndarray, emb: np.ndarray, p: Dict[str, np.ndarray]) -> np.ndarray:
        box_feats, emb = np.atleast_2d(box_feats), np.atleast_2d(emb)
        z = (emb - p["pca_mean"]) @ p["pca_components"].T
        return (np.hstack([box_feats, z]) - p["x_mean"]) / p["x_scale"]

    def raw_predict(self, box_feats: np.ndarray, emb: np.ndarray) -> np.ndarray:
        p = self.params
        return self.build_x(box_feats, emb, p) @ p["coef"] + float(p["intercept"])

    def predict(self, boxes: np.ndarray, frame_h: int = 480, frame_w: int = 640,
                embedding: Optional[np.ndarray] = None) -> int:
        bf = box_features(boxes, frame_h, frame_w).reshape(1, -1)
        if self.params is not None:
            if embedding is None:
                raise ValueError("This count corrector needs the frame embedding")
            value = float(self.raw_predict(bf, embedding.reshape(1, -1))[0])
            a = float(self.params["ema_alpha"])
            self._ema = value if self._ema is None or a >= 1 else a * value + (1 - a) * self._ema
            value = self._ema
        else:
            value = float(self.model.predict(bf)[0])
        return max(0, int(round(value)))

    def save(self, path: str):
        if self.params is not None:
            np.savez(path, **self.params)
        else:
            import joblib
            joblib.dump({"model": self.model, "feature_names": FEATURE_NAMES}, path)

    @classmethod
    def load(cls, path: str) -> Optional["CountCorrector"]:
        if not os.path.exists(path):
            return None
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as d:
                return cls(params={k: d[k] for k in d.files})
        import joblib
        return cls(joblib.load(path)["model"])
