# Crowd Risk Forecasting System

A comprehensive system for detecting and predicting crowd-related risks using computer vision and machine learning.

## Features

- Real-time people detection using YOLOv8
- Crowd density analysis
- Spatio-temporal feature extraction
- Machine learning-based risk prediction
- Anomaly detection
- Web-based dashboard with live video feed
- Alert system for high-risk situations

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
crowd-risk-detection/
├── src/
│   ├── __init__.py
│   ├── detector.py          # YOLOv8 people detection
│   ├── analyzer.py          # Crowd analysis and features
│   ├── predictor.py         # ML models and prediction
│   ├── anomaly_detector.py  # Anomaly detection
│   └── utils.py            # Utility functions
├── models/                  # Trained models
├── data/                   # Sample videos and data
├── api/                    # Flask API
├── frontend/               # React frontend
├── notebooks/              # Jupyter notebooks for analysis
└── tests/                  # Unit tests
```

## Usage

### Basic Detection
```python
from src.detector import PeopleDetector

detector = PeopleDetector()
detector.process_video("path/to/video.mp4")
```

### Risk Prediction
```python
from src.predictor import RiskPredictor

predictor = RiskPredictor()
risk_level = predictor.predict(features)
```

### Web Interface
```bash
python api/app.py
```

Then open `http://localhost:5000` in your browser.

## Phases

1. **Phase 1**: Video processing and people detection
2. **Phase 2**: Crowd density and spatio-temporal analysis
3. **Phase 3**: Machine learning model training
4. **Phase 4**: Anomaly detection
5. **Phase 5**: Web API and frontend
6. **Phase 6**: Optimization and advanced features
