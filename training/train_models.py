"""
Step 2: train all CrowdGuard models on the Mall dataset.

Needs training/cache/detections.npz from extract_detections.py.

Trains, evaluates on a held-out time split, and saves to models/:
  1. count_corrector.joblib   raw YOLO boxes -> calibrated people count
  2. risk_predictor.pkl       Random Forest, low / medium / high risk
  3. anomaly_detector.pkl     Isolation Forest on normal crowd features
  4. lstm/                    LSTM forecasting the next 10 frames' count

Split: Mall is one continuous video, so a random split would leak
near-identical neighbouring frames into the test set. We use the first
80% of frames for training and the last 20% for testing.

Usage:  python training/train_models.py
"""
import glob
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error
from sklearn.model_selection import KFold, cross_val_predict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import logging  # noqa: E402
logging.disable(logging.INFO)

from analyzer import CrowdAnalyzer  # noqa: E402
from count_corrector import FEATURE_NAMES as BOX_FEATURES, box_features  # noqa: E402
from lstm_predictor import LSTMCrowdPredictor, TORCH_AVAILABLE  # noqa: E402
from mall_models import (COUNT_MODELS, fit_anomaly_detector, fit_count_corrector,  # noqa: E402
                         fit_risk_predictor, save_training_data)

MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(os.path.dirname(__file__), "cache", "detections.npz")
CAPACITY, ALERT_RATIO, DISPLAY_CONF = 30, 0.8, 0.15
TRAIN_FRAC = 0.8

# Features the risk and anomaly models use, in a fixed order.
RISK_FEATURES = [
    "people_count", "density_per_frame", "area_coverage", "avg_inter_person_distance",
    "density_variation", "growth_rate", "count_change_rate", "acceleration",
    "density_trend", "volatility", "avg_count", "max_count", "min_count",
    "avg_magnitude", "max_magnitude", "std_magnitude", "movement_entropy",
    "angle_consistency", "movement_concentration",
]


def risk_label(count):
    if count > CAPACITY:
        return "high"
    if count >= CAPACITY * ALERT_RATIO:
        return "medium"
    return "low"


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def dump(r):
    return json.dumps(r, indent=2, default=_np)


def section(title):
    print("\n" + "=" * 64 + f"\n{title}\n" + "=" * 64)


def main():
    os.makedirs(MODELS, exist_ok=True)
    report = {}

    gt = np.array(sio.loadmat(os.path.join(ROOT, "data/mall_gt.mat"))["count"]).reshape(-1).astype(float)
    frames = sorted(glob.glob(os.path.join(ROOT, "data/frames/frames/*.jpg")))
    d = np.load(CACHE)
    boxes, n = d["boxes"], int(d["n_frames"])
    assert n == len(gt) == len(frames), (n, len(gt), len(frames))
    per_frame = [boxes[boxes[:, 0] == i, 1:] for i in range(n)]
    split = int(n * TRAIN_FRAC)
    tr, te = slice(0, split), slice(split, n)
    print(f"{n} frames: train 1..{split}, test {split + 1}..{n}")

    # ------------------------------------------------------------------
    section("1. Count corrector")
    X = np.stack([box_features(b) for b in per_frame])
    raw = X[:, BOX_FEATURES.index("n_conf_0.15")]
    # choose the model on the last 20% of the training part, report on test
    val = slice(int(split * 0.8), split)
    fit = slice(0, int(split * 0.8))
    candidates = COUNT_MODELS
    val_mae = {}
    for name, make in candidates.items():
        m = make().fit(X[fit], gt[fit])
        val_mae[name] = mean_absolute_error(gt[val], m.predict(X[val]))
    best = min(val_mae, key=val_mae.get)
    cc = fit_count_corrector(X[tr], gt[tr], best)
    corrected_test = np.clip(np.round(cc.model.predict(X[te])), 0, None)
    # out-of-fold predictions for training frames, so the downstream models
    # see realistic (not memorised) counts
    oof = cross_val_predict(candidates[best](), X[tr], gt[tr], cv=KFold(5, shuffle=False))
    corrected = np.concatenate([np.clip(np.round(oof), 0, None), corrected_test])

    r = {
        "model": best,
        "validation_mae": {k: round(v, 2) for k, v in val_mae.items()},
        "test_mae_raw_yolo": round(mean_absolute_error(gt[te], raw[te]), 2),
        "test_mae_corrected": round(mean_absolute_error(gt[te], corrected_test), 2),
        "test_mean_gt": round(gt[te].mean(), 1),
        "test_mean_raw": round(raw[te].mean(), 1),
        "test_mean_corrected": round(corrected_test.mean(), 1),
    }
    lvl = lambda a: np.array([risk_label(c) for c in a])  # noqa: E731
    r["test_alert_level_match_raw"] = round(float(np.mean(lvl(raw[te]) == lvl(gt[te]))), 3)
    r["test_alert_level_match_corrected"] = round(float(np.mean(lvl(corrected_test) == lvl(gt[te]))), 3)
    print(dump(r))
    report["count_corrector"] = r
    cc.save(os.path.join(MODELS, "count_corrector.joblib"))

    # ------------------------------------------------------------------
    section("Replaying frames through the analyzer (optical flow etc.)")
    an = CrowdAnalyzer(frame_shape=(480, 640), history_length=30)
    rows = []
    for i, path in enumerate(frames):
        b = per_frame[i]
        b = b[b[:, 4] >= DISPLAY_CONF]
        dets = [{"bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": float(c),
                 "center": [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                 "area": int((x2 - x1) * (y2 - y1))} for x1, y1, x2, y2, c in b]
        f = an.update_frame(cv2.imread(path), dets, people_count=int(corrected[i]))
        rows.append({k: f.get(k, 0.0) for k in RISK_FEATURES})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{n}")
    feats = pd.DataFrame(rows, columns=RISK_FEATURES).fillna(0.0).astype(float)
    feats.to_csv(os.path.join(os.path.dirname(__file__), "cache", "mall_features.csv"), index=False)

    # ------------------------------------------------------------------
    section("2. Risk predictor (Random Forest)")
    y_lab = np.array([risk_label(c) for c in gt])
    rp = fit_risk_predictor(feats.values[tr], y_lab[tr], RISK_FEATURES)
    pred = rp.label_encoder.inverse_transform(rp.model.predict(rp.scaler.transform(feats.values[te])))
    rule = lvl(corrected_test)
    labels = ["low", "medium", "high"]
    imp = sorted(zip(RISK_FEATURES, rp.model.feature_importances_), key=lambda t: -t[1])[:6]
    r = {
        "test_accuracy": round(accuracy_score(y_lab[te], pred), 3),
        "test_macro_f1": round(f1_score(y_lab[te], pred, average="macro"), 3),
        "rule_on_corrected_count_accuracy": round(accuracy_score(y_lab[te], rule), 3),
        "confusion_matrix_rows_true_cols_pred": {"labels": labels,
                                                  "matrix": confusion_matrix(y_lab[te], pred, labels=labels).tolist()},
        "test_class_counts": {k: int(np.sum(y_lab[te] == k)) for k in labels},
        "top_features": {k: round(float(v), 3) for k, v in imp},
    }
    print(dump(r))
    report["risk_predictor"] = r
    rp.save_model(os.path.join(MODELS, "risk_predictor.pkl"))

    # ------------------------------------------------------------------
    section("3. Anomaly detector (Isolation Forest)")
    ad = fit_anomaly_detector(feats.values[tr], RISK_FEATURES)
    xs_te = ad.scaler.transform(feats.values[te])
    dec = ad.model.decision_function(xs_te)
    norm = np.clip(-dec / ad.score_scale, 0, 1)
    flagged = ad.model.predict(xs_te) == -1
    top = np.argsort(-norm)[:5]
    r = {
        "train_anomaly_rate": 0.05,
        "test_anomaly_rate": round(float(flagged.mean()), 3),
        "test_frames_over_alert_threshold_0.6": int(np.sum(flagged & (norm > 0.6))),
        "score_scale": round(ad.score_scale, 4),
        "most_anomalous_test_frames": [
            {"frame": int(split + i + 1), "score": round(float(norm[i]), 2), "gt_count": int(gt[split + i]),
             "growth_rate": round(float(feats.growth_rate.iloc[split + i]), 2),
             "avg_motion": round(float(feats.avg_magnitude.iloc[split + i]), 2)} for i in top],
    }
    print(dump(r))
    report["anomaly_detector"] = r
    ad.save_model(os.path.join(MODELS, "anomaly_detector.pkl"))
    # Ship the training data so other scikit-learn versions can refit
    save_training_data(MODELS, X, gt, feats.values, y_lab, RISK_FEATURES, split, best)

    # ------------------------------------------------------------------
    section("4. LSTM forecaster (next 10 frames)")
    report_path = os.path.join(MODELS, "training_report.json")
    if not TORCH_AVAILABLE:
        print("PyTorch not installed: skipping LSTM, keeping the existing models/lstm")
        if os.path.exists(report_path):
            report["lstm_forecaster"] = json.load(open(report_path)).get("lstm_forecaster")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=_np)
        return
    series = np.stack([corrected, feats.density_per_frame, feats.avg_magnitude,
                       feats.movement_concentration], axis=1).astype(np.float32)
    # Small network + few epochs: bigger ones overfit the 1600 training
    # frames and lose to the "count stays the same" baseline.
    import torch
    torch.manual_seed(1)
    np.random.seed(1)
    lstm = LSTMCrowdPredictor(window_size=30, prediction_steps=10, hidden_size=16)
    res = lstm.train_on_sequence(series[tr], epochs=40, batch_size=32, lr=0.001)
    # evaluate on test windows against the true (GT) future counts
    Xt, _ = lstm.make_windows(series[split - 30:])  # include 30 frames of context
    yt = np.array([gt[split - 30 + i + 30: split - 30 + i + 40] for i in range(len(Xt))])
    preds = np.array([lstm.predict(x)["predictions"] for x in Xt])
    persist = np.repeat(Xt[:, -1, 0:1], 10, axis=1)          # "count stays the same"
    window_mean = np.repeat(Xt[:, :, 0].mean(1, keepdims=True), 10, axis=1)
    r = {
        "train_status": res.get("status"),
        "best_val_loss": round(float(res.get("best_val_loss", 0)), 4),
        "test_windows": int(len(Xt)),
        "test_mae_lstm": round(float(np.abs(preds - yt).mean()), 2),
        "test_mae_persistence_baseline": round(float(np.abs(persist - yt).mean()), 2),
        "test_mae_window_mean_baseline": round(float(np.abs(window_mean - yt).mean()), 2),
        "test_mae_by_horizon_lstm": [round(float(v), 2) for v in np.abs(preds - yt).mean(0)],
    }
    print(dump(r))
    report["lstm_forecaster"] = r
    lstm.save(os.path.join(MODELS, "lstm"))

    with open(os.path.join(MODELS, "training_report.json"), "w") as f:
        json.dump(report, f, indent=2, default=_np)
    print(f"\nSaved models and training_report.json to {MODELS}")


if __name__ == "__main__":
    main()
