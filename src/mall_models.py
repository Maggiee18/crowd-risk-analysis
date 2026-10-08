"""
Fitting and version-safe loading for the Mall-trained scikit-learn models.

scikit-learn pickles are not portable across versions: a Random Forest
saved with 1.3 loads in 1.9 but returns nonsense probabilities (e.g. 63.5
instead of 0..1). So models/ also ships the (small) training data, and if
the installed scikit-learn/numpy differ from the ones the pickles were made
with, the models are refit in memory at startup. That takes ~2 seconds.

Used by training/train_models.py (fitting) and by the API / main.py (loading).
"""
import json
import logging
import os
from typing import Dict

import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler

from anomaly_detector import AnomalyDetector
from count_corrector import CountCorrector
from predictor import RiskPredictor

logger = logging.getLogger(__name__)

DATA_FILE = "mall_training_data.npz"
VERSIONS_FILE = "versions.json"

COUNT_MODELS = {
    "ridge": lambda: Ridge(alpha=1.0),
    "gradient_boosting": lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=42),
}


def current_versions() -> Dict[str, str]:
    return {"sklearn": sklearn.__version__, "numpy": np.__version__}


def _major_minor(v: str) -> str:
    return ".".join(v.split(".")[:2])


def fit_count_corrector(X_box, gt, model_name="ridge") -> CountCorrector:
    return CountCorrector(COUNT_MODELS[model_name]().fit(X_box, gt))


def fit_risk_predictor(feats, labels, feature_names) -> RiskPredictor:
    rp = RiskPredictor(model_type="random_forest")
    rp.feature_names = list(feature_names)
    rp.label_encoder = LabelEncoder().fit(["high", "low", "medium"])
    rp.scaler = StandardScaler().fit(feats)
    rp.model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                      random_state=42, n_jobs=-1)
    rp.model.fit(rp.scaler.transform(feats), rp.label_encoder.transform(labels))
    rp.is_trained = True
    return rp


def fit_anomaly_detector(feats, feature_names) -> AnomalyDetector:
    ad = AnomalyDetector(method="isolation_forest", contamination=0.05)
    ad.feature_names = list(feature_names)
    ad.scaler = StandardScaler().fit(feats)
    xs = ad.scaler.transform(feats)
    ad.model = IsolationForest(contamination=0.05, n_estimators=200, random_state=42).fit(xs)
    ad.is_trained = True
    ad.score_scale = float(-ad.model.decision_function(xs).min())
    return ad


def save_training_data(models_dir, X_box, gt, feats, labels, feature_names, split, count_model):
    np.savez_compressed(
        os.path.join(models_dir, DATA_FILE),
        X_box=X_box.astype(np.float32), gt=gt.astype(np.float32), feats=feats.astype(np.float32),
        labels=np.array(labels), feature_names=np.array(feature_names),
        split=split, count_model=count_model)
    with open(os.path.join(models_dir, VERSIONS_FILE), "w") as f:
        json.dump(current_versions(), f, indent=2)


def _refit_all(models_dir) -> Dict:
    d = np.load(os.path.join(models_dir, DATA_FILE), allow_pickle=False)
    s = int(d["split"])
    names = [str(n) for n in d["feature_names"]]
    return {
        "count_corrector": fit_count_corrector(d["X_box"][:s], d["gt"][:s], str(d["count_model"])),
        "risk_predictor": fit_risk_predictor(d["feats"][:s], d["labels"][:s], names),
        "anomaly_detector": fit_anomaly_detector(d["feats"][:s], names),
    }


def load_mall_models(models_dir: str) -> Dict:
    """
    Returns {'count_corrector', 'risk_predictor', 'anomaly_detector'} (any may be
    missing). Loads the pickles when the library versions match, otherwise
    refits from the shipped training data.
    """
    out = {}
    try:
        with open(os.path.join(models_dir, VERSIONS_FILE)) as f:
            saved = json.load(f)
    except (OSError, ValueError):
        saved = {}
    now = current_versions()
    same = saved and all(_major_minor(saved.get(k, "")) == _major_minor(v) for k, v in now.items())

    if same:
        try:
            cc = CountCorrector.load(os.path.join(models_dir, "count_corrector.joblib"))
            if cc is not None:
                out["count_corrector"] = cc
            for key, obj, fname in [("risk_predictor", RiskPredictor(), "risk_predictor.pkl"),
                                    ("anomaly_detector", AnomalyDetector(), "anomaly_detector.pkl")]:
                path = os.path.join(models_dir, fname)
                if os.path.exists(path):
                    obj.load_model(path)
                    out[key] = obj
            return out
        except Exception as e:
            logger.warning(f"Could not load saved models ({e}), refitting")

    if os.path.exists(os.path.join(models_dir, DATA_FILE)):
        logger.warning(f"Models were saved with {saved or 'unknown versions'}, running {now}: "
                       "refitting from models/mall_training_data.npz")
        return _refit_all(models_dir)
    return out
