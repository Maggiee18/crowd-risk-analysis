# 🛡️ CrowdGuard — AI-Powered Crowd Risk Prediction & Intelligent Control System

A production-ready, real-time AI system for crowd detection, risk prediction, behavior analysis, and automated control suggestions.

## ⚡ Features

| Feature | Technology |
|---------|-----------|
| **People Detection** | YOLOv8 (Ultralytics) |
| **Multi-Object Tracking** | Deep SORT / Centroid tracker |
| **Crowd Density** | Spatial density mapping |
| **Behavior Analysis** | Dense optical flow (Farneback) |
| **Risk Prediction** | Random Forest / SVM / Ensemble |
| **Anomaly Detection** | Isolation Forest / One-Class SVM |
| **Future Prediction** | LSTM temporal model (PyTorch) |
| **Control Suggestions** | Q-learning RL agent |
| **Face Privacy** | Haar cascade face blurring |
| **Live Dashboard** | React + Tailwind + Recharts |
| **Real-time Streaming** | FastAPI + WebSocket |
| **Simulation Mode** | 6 crowd scenarios |
| **Admin Auth** | Token-based login |
| **Incident Reports** | Automated HTML Snapshot Gen |
| **Natural Language AI** | Pattern-matching NL query |
| **Sentiment Estimation** | Multi-factor heuristic mapping |
| **Audio Alert System** | Web Audio API synthetic tones |
| **Geofence Zone Editor** | HTML Canvas interactive map |

## 📁 Project Structure

```
crowdguard/
├── src/
│   ├── detector.py           # YOLOv8 people detection
│   ├── analyzer.py           # Crowd analysis & optical flow
│   ├── predictor.py          # ML risk prediction
│   ├── anomaly_detector.py   # Anomaly detection
│   ├── optimizer.py          # Real-time processing & alerts
│   ├── tracker.py            # Deep SORT multi-object tracking
│   ├── lstm_predictor.py     # LSTM temporal prediction
│   ├── rl_controller.py      # RL crowd control agent
│   ├── privacy.py            # Face blurring
│   ├── simulator.py          # Mock crowd scenarios
│   └── utils.py              # Utilities
├── api/
│   ├── main.py               # FastAPI + WebSocket server
│   └── app.py                # Legacy Flask API
├── frontend/
│   ├── src/
│   │   ├── App.jsx           # Main dashboard
│   │   ├── components/       # 10+ UI components
│   │   ├── hooks/            # WebSocket hook
│   │   └── services/         # API service
│   ├── tailwind.config.js
│   └── package.json
├── models/                   # Trained ML models
├── data/                     # Sample data & videos
├── config.json               # System configuration
├── requirements.txt          # Python dependencies
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## 🚀 Quick Start

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Backend (FastAPI)

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
- API docs: `http://localhost:8000/docs`
- WebSocket: `ws://localhost:8000/ws`

### 3. Start Frontend (React)

```bash
cd frontend
npm install
npm start
```

Dashboard available at `http://localhost:3000`.

**Login:** `admin` / `admin123`

### 4. Run Simulation

You can start a simulation from the dashboard or via API:

```bash
curl -X POST http://localhost:8000/api/simulate -H "Content-Type: application/json" -d '{"scenario": "sudden_surge"}'
```

## 🧠 Training on the Mall Dataset

All models are trained on the 2000 Mall frames with `data/mall_gt.mat` as ground truth. The first 80% of frames are used for training and the last 20% for testing (a random split would leak near identical neighbouring frames into the test set).

```bash
python training/extract_detections.py   # YOLO boxes + image features for all frames, cached (~20 min on CPU)
python training/train_models.py         # trains + evaluates all models, saves to models/
```

The API and `main.py` load whatever is in `models/` automatically on startup.

| Model | File | Test result (last 400 frames) |
|---|---|---|
| Count corrector v2 (YOLO boxes + YOLO backbone image features, ridge, light smoothing) | `count_corrector.npz` | Count error 4.34 (raw YOLO) → 2.42 (boxes only) → **1.71** people per frame; alert level matches ground truth 57% → **83%** |
| Risk predictor (Random Forest) | `risk_predictor.pkl` | **86%** accuracy, macro F1 0.867 |
| Anomaly detector (Isolation Forest) | `anomaly_detector.pkl` | Flags 1.5% of test frames; scores calibrated so 1.0 = most unusual training frame |
| LSTM forecaster (3 model ensemble, next 10 frames) | `lstm/` | Error **3.44** people vs 3.52 for "count stays the same" |

All settings (PCA size, ridge strength, smoothing, LSTM blend) are chosen on a validation slice (frames 1281 to 1600), never on the test frames.

Notes:
- Risk labels come from the ground truth count (low < 24, medium 24 to 30, high > 30), so the Random Forest cannot do much better than the count itself. To learn real risk, it needs labelled incidents.
- The Mall video has no real incidents, so the anomaly detector mostly confirms normal behaviour.
- The scikit-learn pickles were saved with scikit-learn 1.3 / numpy 1.24 (the pinned versions). With any other version they are refit automatically at startup from `models/mall_training_data.npz` (about 1 second), because pickles across versions give wrong probabilities.
- Full metrics are in `models/training_report.json`.

## 🏃 Panic Detection

`src/panic_detector.py` raises a **CRITICAL panic alert** when a crowd suddenly moves in an unusual way. It needs no training and adapts to each camera:

- Speeds are measured in **body heights per frame**, so camera resolution, distance and computer speed don't matter.
- It learns the crowd's own "normal" movement first, then flags panic when at least 40% of people move faster than 1.5x the fast end of normal (the 90th percentile of normal speeds) for 3 frames in a row.
- Each person must be fast on **two frames in a row**, so tracker mix-ups (ID swaps) don't count as running.
- Scattered directions are reported as `scatter` (panic), and everyone running the same way as `rush`. Both raise a critical alert and force the risk level to high.

Results: the Panic Event simulation is caught on 98% of its panic frames, within about 4 to 6 frames (under half a second). There are **zero false alarms** on all 2000 real Mall frames and in the calm simulations. Tests are in `tests/test_panic_detector.py`.

## 🐳 Docker

```bash
docker-compose up --build
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/status` | System health & components |
| GET | `/api/current_state` | Latest analysis results |
| POST | `/api/process_frame` | Process base64 frame |
| GET | `/api/alerts` | Active alerts |
| GET | `/api/history` | Historical data |
| GET | `/api/predictions` | LSTM predictions |
| GET | `/api/control_suggestions` | RL suggestions |
| GET | `/api/zones` | Zone-based analysis |
| GET | `/api/scenarios` | Available simulations |
| POST | `/api/simulate` | Start simulation |
| POST | `/api/simulate/stop` | Stop simulation |
| POST | `/api/auth/login` | Admin login |
| GET | `/api/privacy/toggle` | Toggle face blur |
| GET | `/api/logs/download` | Download logs JSON |
| WS | `/ws` | Real-time WebSocket |

## 🧪 Sample Test Scenarios

Available simulation scenarios:
- `normal` — Steady low-to-moderate density
- `gradual_buildup` — Slowly increasing crowd
- `sudden_surge` — Abrupt crowd spike
- `panic_event` — Normal → panic → return
- `evacuation` — Large crowd dispersing
- `concert` — Waves of density

## 📊 Dashboard Features

- **Live Video Feed** with bounding boxes
- **Risk Explainability Panel** (shows exact factors driving risk scores)
- **Sentiment Gauge** (estimates crowd mood from Calming to Panic)
- **Real-time Charts** (people count, density, risk trends)
- **LSTM Prediction & Comparative Analytics** (future crowd forecast & split views)
- **Interactive Zone Editor** (draw bounding boxes for specific zone risk)
- **Flow Direction Arrows** (visualize crowd movement physics)
- **Alert Panel & Audio Sirens** (severity-coded notifications with procedural tone generation)
- **Session Timeline** (visual history of anomaly/risk changes)
- **NL Query Panel** (Ask "What is the peak crowd?" via built-in chatbot)
- **Emergency Lockdown Broadcast** (instant data capture & alert sequence)
- **Interactive What-If Simulator** (adjust sliders to test AI risk response)
- **Privacy Toggle** (face blurring on/off)
- **1-Click Incident Reports** (download HTML snapshots instantly)

## ⚙️ Configuration

Edit `config.json` or use environment variables (see `.env.example`).

## 📄 License

Open source — all libraries used are open-source.
