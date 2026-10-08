"""
Step 2: train all CrowdGuard models on the Mall dataset.

Needs training/cache/detections.npz from extract_detections.py.

Trains, evaluates on a held-out time split, and saves to models/:
  1. count_corrector.npz      YOLO boxes + backbone image features -> people count
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

import logging  # noqa: E402
logging.disable(logging.INFO)

from analyzer import CrowdAnalyzer  # noqa: E402
from count_corrector import CountCorrector, FEATURE_NAMES as BOX_FEATURES, box_features  # noqa: E402
from lstm_predictor import LSTMCrowdPredictor, TORCH_AVAILABLE  # noqa: E402
from mall_models import fit_anomaly_detector, fit_risk_predictor, save_training_data  # noqa: E402

MODELS = os.path.join(ROOT, "models")
CACHE = os.path.join(os.path.dirname(__file__), "cache", "detections.npz")
EMB_CACHE = os.path.join(os.path.dirname(__file__), "cache", "embeddings.npz")
PCA_SIZES = (32, 64, 96, 128)
RIDGE_ALPHAS = (0.1, 1.0, 10.0)
EMA_ALPHAS = (1.0, 0.7, 0.5)
LSTM_SEEDS = (1, 2, 3)
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


def ema(p, a):
    """Causal exponential smoothing (a=1 means no smoothing)."""
    p = np.asarray(p, dtype=float)
    if a >= 1:
        return p.copy()
    out = p.copy()
    for i in range(1, len(p)):
        out[i] = a * p[i] + (1 - a) * out[i - 1]
    return out


def fit_corrector_params(Xb, E, y, k, alpha, ema_alpha):
    """PCA on image embeddings + standardise + ridge, returned as numpy arrays."""
    pca = PCA(k, random_state=0).fit(E)
    Z = np.hstack([Xb, pca.transform(E)])
    sc = StandardScaler().fit(Z)
    rg = Ridge(alpha=alpha).fit(sc.transform(Z), y)
    return {"pca_mean": pca.mean_, "pca_components": pca.components_,
            "x_mean": sc.mean_, "x_scale": sc.scale_, "coef": rg.coef_,
            "intercept": np.array(rg.intercept_), "ema_alpha": np.array(ema_alpha)}


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
    section("1. Count corrector (box + image features)")
    X = np.stack([box_features(b) for b in per_frame])
    emb = np.load(EMB_CACHE)["emb"].astype(np.float64)
    raw = X[:, BOX_FEATURES.index("n_conf_0.15")]
    # choose settings on the last 20% of the training part, report on test
    v = int(split * 0.8)
    v1 = Ridge(alpha=1.0).fit(X[:v], gt[:v])
    v1_val = mean_absolute_error(gt[v:split], np.round(v1.predict(X[v:split])))
    grid = []
    for k in PCA_SIZES:
        for alpha in RIDGE_ALPHAS:
            p_ = fit_corrector_params(X[:v], emb[:v], gt[:v], k, alpha, 1.0)
            raw_pred = CountCorrector(params=p_).raw_predict(X, emb)
            for e in EMA_ALPHAS:
                grid.append((mean_absolute_error(gt[v:split], np.round(ema(raw_pred, e)[v:split])), k, alpha, e))
    grid.sort()
    val_mae, k, alpha, e = grid[0]
    print(f"validation: box-only ridge {v1_val:.2f}, best v2 {val_mae:.2f} (pca={k}, alpha={alpha}, ema={e})")

    params = fit_corrector_params(X[tr], emb[tr], gt[tr], k, alpha, e)
    cc = CountCorrector(params=params)
    corrected_test = np.clip(np.round(ema(cc.raw_predict(X, emb), e)[te]), 0, None)
    # out-of-fold predictions for training frames (5 contiguous folds), so
    # the downstream models see realistic (not memorised) counts
    oof = np.zeros(split)
    for fold in np.array_split(np.arange(split), 5):
        mask = np.ones(split, bool)
        mask[fold] = False
        pf = fit_corrector_params(X[:split][mask], emb[:split][mask], gt[:split][mask], k, alpha, e)
        oof[fold] = CountCorrector(params=pf).raw_predict(X[fold], emb[fold])
    corrected = np.concatenate([np.clip(np.round(ema(oof, e)), 0, None), corrected_test])

    v1_full = Ridge(alpha=1.0).fit(X[tr], gt[tr])
    r = {
        "model": "v2: ridge on box features + PCA of YOLO backbone features, EMA smoothed",
        "settings": {"pca_components": int(k), "ridge_alpha": alpha, "ema_alpha": e},
        "validation_mae": {"box_only_v1": round(v1_val, 2), "v2": round(val_mae, 2)},
        "test_mae_raw_yolo": round(mean_absolute_error(gt[te], raw[te]), 2),
        "test_mae_box_only_v1": round(mean_absolute_error(gt[te], np.round(v1_full.predict(X[te]))), 2),
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
    cc.save(os.path.join(MODELS, "count_corrector.npz"))
    old = os.path.join(MODELS, "count_corrector.joblib")
    if os.path.exists(old):
        os.remove(old)

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
    save_training_data(MODELS, gt, feats.values, y_lab, RISK_FEATURES, split)

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
    # frames. Three seeds are averaged (single runs vary a lot), then blended
    # with "count stays the same" using a weight chosen on validation.
    import torch

    def train_ensemble(end):
        members = []
        for sd in LSTM_SEEDS:
            torch.manual_seed(sd)
            np.random.seed(sd)
            m = LSTMCrowdPredictor(window_size=30, prediction_steps=10, hidden_size=16)
            res_ = m.train_on_sequence(series[:end], epochs=40, batch_size=32, lr=0.001)
            members.append((m, res_))
        main_m = members[0][0]
        main_m.extra_models = [m.model for m, _ in members[1:]]
        return main_m, members[0][1]

    def windows(start, end):
        Xw, _ = lstm.make_windows(series[start - 30:end])
        yw = np.array([gt[start + i:start + i + 10] for i in range(len(Xw))])
        return Xw, yw

    v_l = int(split * 0.8)
    lstm, _ = train_ensemble(v_l)
    Xv, yv = windows(v_l + 30, split)
    pv = np.array([lstm.predict(x)["predictions"] for x in Xv])
    pers_v = np.repeat(Xv[:, -1, 0:1], 10, axis=1)
    weights = np.round(np.linspace(0, 1, 11), 1)
    blend = float(weights[np.argmin([np.abs(w * pv + (1 - w) * pers_v - yv).mean() for w in weights])])

    lstm, res = train_ensemble(split)
    Xt, yt = windows(split, n)
    lstm.blend_weight = 1.0
    pure = np.array([lstm.predict(x)["predictions"] for x in Xt])
    lstm.blend_weight = blend
    preds = np.array([lstm.predict(x)["predictions"] for x in Xt])
    persist = np.repeat(Xt[:, -1, 0:1], 10, axis=1)          # "count stays the same"
    window_mean = np.repeat(Xt[:, :, 0].mean(1, keepdims=True), 10, axis=1)
    r = {
        "train_status": res.get("status"),
        "members": len(LSTM_SEEDS),
        "blend_weight_chosen_on_validation": blend,
        "test_windows": int(len(Xt)),
        "test_mae_lstm": round(float(np.abs(preds - yt).mean()), 2),
        "test_mae_lstm_ensemble_unblended": round(float(np.abs(pure - yt).mean()), 2),
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
