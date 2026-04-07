"""
Flask API for Crowd Risk Detection System
Provides REST endpoints for real-time crowd analysis and risk prediction
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
import threading
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PeopleDetector
from analyzer import CrowdAnalyzer
from predictor import RiskPredictor, EnsemblePredictor
from anomaly_detector import AnomalyDetector
from optimizer import AlertSystem
from utils import save_json, load_json, get_timestamp

app = Flask(__name__)
CORS(app)

# Global variables for system components
detector = None
analyzer = None
risk_predictor = None
anomaly_detector = None
ensemble_predictor = None
alert_system = None
current_frame = None
current_detections = []
current_features = {}
current_alerts = []
system_status = {
    'initialized': False,
    'processing': False,
    'total_frames_processed': 0,
    'start_time': None,
    'last_update': None
}

# Thread lock for thread safety
processing_lock = threading.Lock()

def initialize_system():
    """Initialize all system components"""
    global detector, analyzer, risk_predictor, anomaly_detector, ensemble_predictor, alert_system, system_status
    
    try:
        # Initialize components
        detector = PeopleDetector()
        analyzer = CrowdAnalyzer(frame_shape=(480, 640))
        risk_predictor = RiskPredictor(model_type='random_forest')
        anomaly_detector = AnomalyDetector(method='isolation_forest')
        ensemble_predictor = EnsemblePredictor()
        alert_system = AlertSystem(
            risk_threshold=0.7,
            anomaly_threshold=0.6,
            capacity_limit=30,
            alert_ratio=0.8,
            trend_min_slope=0.5
        )
        
        # Try to load pre-trained models
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        risk_model_file = os.path.join(model_path, 'risk_predictor.pkl')
        if os.path.exists(risk_model_file):
            risk_predictor.load_model(risk_model_file)
            print("Loaded pre-trained risk prediction model")
        
        anomaly_model_file = os.path.join(model_path, 'anomaly_detector.pkl')
        if os.path.exists(anomaly_model_file):
            anomaly_detector.load_model(anomaly_model_file)
            print("Loaded pre-trained anomaly detection model")
        
        system_status['initialized'] = True
        system_status['start_time'] = datetime.now().isoformat()
        print("System initialized successfully")
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        system_status['initialized'] = False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'ok',
        'system': system_status,
        'components': {
            'detector': detector is not None,
            'analyzer': analyzer is not None,
            'risk_predictor': risk_predictor is not None,
            'anomaly_detector': anomaly_detector is not None
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict_risk():
    """
    Predict crowd risk from features
    Expects JSON with features array
    """
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Features required'}), 400
        
        features = np.array(data['features'])
        
        if risk_predictor is None or not risk_predictor.is_trained:
            return jsonify({'error': 'Risk predictor not trained'}), 400
        
        # Make prediction
        risk_level, confidence = risk_predictor.predict(features)
        
        # Anomaly detection
        anomaly_result = {}
        if anomaly_detector and anomaly_detector.is_trained:
            is_anomaly, anomaly_score, anomaly_details = anomaly_detector.detect_anomaly(features)
            anomaly_result = {
                'is_anomaly': is_anomaly,
                'anomaly_score': anomaly_score,
                'details': anomaly_details
            }
        
        response = {
            'risk_level': risk_level,
            'confidence': confidence,
            'anomaly_detection': anomaly_result,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/process_frame', methods=['POST'])
def process_frame():
    """
    Process a single frame for people detection and analysis
    Expects base64 encoded image
    """
    global current_frame, current_detections, current_features, current_alerts, system_status
    
    try:
        with processing_lock:
            data = request.get_json()
            
            if not data or 'image' not in data:
                return jsonify({'error': 'Image data required'}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({'error': 'Invalid image data'}), 400
            
            # Resize frame if needed
            if frame.shape[:2] != (480, 640):
                frame = cv2.resize(frame, (640, 480))
            
            # Detect people
            detections, count = detector.detect_people(frame)
            
            # Analyze crowd
            features = analyzer.update_frame(frame, detections)
            
            # Risk prediction
            risk_result = {}
            if risk_predictor and risk_predictor.is_trained:
                feature_vector = analyzer.get_feature_vector()
                if len(feature_vector) > 0:
                    risk_level, confidence = risk_predictor.predict(feature_vector)
                    risk_result = {
                        'risk_level': risk_level,
                        'confidence': confidence
                    }
            
            # Anomaly detection
            anomaly_result = {}
            if anomaly_detector and anomaly_detector.is_trained:
                feature_vector = analyzer.get_feature_vector()
                if len(feature_vector) > 0:
                    is_anomaly, anomaly_score, anomaly_details = anomaly_detector.detect_anomaly(feature_vector)
                    anomaly_result = {
                        'is_anomaly': is_anomaly,
                        'anomaly_score': anomaly_score,
                        'details': anomaly_details
                    }
            
            # Update global state
            current_frame = frame.copy()
            current_detections = detections
            current_features = features
            system_status['total_frames_processed'] += 1
            system_status['last_update'] = datetime.now().isoformat()

            # Alerts
            alerts = []
            if alert_system:
                alerts = alert_system.evaluate_risk(risk_result, anomaly_result, features)
            current_alerts = alerts
            
            # Prepare response
            response = {
                'people_count': count,
                'detections': detections,
                'features': {
                    'people_count': features.get('people_count', 0),
                    'density_per_frame': features.get('density_per_frame', 0),
                    'growth_rate': features.get('growth_rate', 0),
                    'volatility': features.get('volatility', 0),
                    'avg_magnitude': features.get('avg_magnitude', 0),
                    'movement_entropy': features.get('movement_entropy', 0)
                },
                'risk_prediction': risk_result,
                'anomaly_detection': anomaly_result,
                'alerts': alerts,
                'timestamp': datetime.now().isoformat()
            }
            
            return jsonify(response)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/video_feed', methods=['POST'])
def process_video_feed():
    """
    Process video feed for real-time analysis
    Expects video file or base64 encoded frames
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Handle different input types
        if 'video_path' in data:
            # Process video file
            video_path = data['video_path']
            if not os.path.exists(video_path):
                return jsonify({'error': 'Video file not found'}), 404
            
            results = process_video_file(video_path)
            return jsonify(results)
        
        elif 'frames' in data:
            # Process base64 encoded frames
            frames_data = data['frames']
            results = process_base64_frames(frames_data)
            return jsonify(results)
        
        else:
            return jsonify({'error': 'Invalid input format'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_video_file(video_path: str) -> Dict:
    """Process video file and return results"""
    global system_status
    
    system_status['processing'] = True
    start_time = time.time()
    
    try:
        # Process video with detector
        results = detector.process_video(video_path, show_live=False)
        
        # Analyze each frame
        all_features = []
        risk_predictions = []
        anomaly_detections = []
        
        for frame_info in results['frame_data']:
            # Simulate frame processing (in real implementation, would process actual frames)
            features = {
                'people_count': frame_info['people_count'],
                'density_per_frame': frame_info['people_count'] * 0.1,  # Simplified
                'growth_rate': 0,
                'volatility': 0.5
            }
            all_features.append(features)
        
        processing_time = time.time() - start_time
        system_status['processing'] = False
        
        return {
            'video_stats': results,
            'processing_time': processing_time,
            'frames_analyzed': len(all_features),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        system_status['processing'] = False
        raise e

def process_base64_frames(frames_data: list) -> Dict:
    """Process base64 encoded frames"""
    results = []
    
    for i, frame_data in enumerate(frames_data):
        try:
            # Decode frame
            image_data = base64.b64decode(frame_data)
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue
            
            # Process frame
            detections, count = detector.detect_people(frame)
            features = analyzer.update_frame(frame, detections)
            
            results.append({
                'frame_index': i,
                'people_count': count,
                'features': features
            })
            
        except Exception as e:
            print(f"Error processing frame {i}: {e}")
            continue
    
    return {
        'frames_processed': len(results),
        'results': results,
        'timestamp': datetime.now().isoformat()
    }

@app.route('/api/current_state', methods=['GET'])
def get_current_state():
    """Get current system state and latest analysis results"""
    global current_frame, current_detections, current_features, system_status
    
    # Encode current frame if available
    frame_data = None
    if current_frame is not None:
        _, buffer = cv2.imencode('.jpg', current_frame)
        frame_data = base64.b64encode(buffer).decode('utf-8')
    
    # Risk prediction
    risk_result = {}
    if risk_predictor and risk_predictor.is_trained and len(analyzer.get_feature_vector()) > 0:
        feature_vector = analyzer.get_feature_vector()
        risk_level, confidence = risk_predictor.predict(feature_vector)
        risk_result = {
            'risk_level': risk_level,
            'confidence': confidence
        }
    
    # Anomaly detection
    anomaly_result = {}
    if anomaly_detector and anomaly_detector.is_trained and len(analyzer.get_feature_vector()) > 0:
        feature_vector = analyzer.get_feature_vector()
        is_anomaly, anomaly_score, anomaly_details = anomaly_detector.detect_anomaly(feature_vector)
        anomaly_result = {
            'is_anomaly': is_anomaly,
            'anomaly_score': anomaly_score,
            'details': anomaly_details
        }
    
    return jsonify({
        'system_status': system_status,
        'current_frame': frame_data,
        'people_count': len(current_detections),
        'detections': current_detections,
        'features': current_features,
        'risk_prediction': risk_result,
        'anomaly_detection': anomaly_result,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """Train risk prediction model with provided data"""
    try:
        data = request.get_json()
        
        if not data or 'features' not in data:
            return jsonify({'error': 'Training data required'}), 400
        
        features_df = pd.DataFrame(data['features'])
        
        # Prepare and train model
        X_train, X_test, y_train, y_test = risk_predictor.prepare_data(features_df)
        training_results = risk_predictor.train(X_train, y_train)
        evaluation_results = risk_predictor.evaluate(X_test, y_test)
        
        # Save model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'risk_predictor.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        risk_predictor.save_model(model_path)
        
        return jsonify({
            'training_results': training_results,
            'evaluation_results': {
                'accuracy': evaluation_results['accuracy'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'f1_score': evaluation_results['f1_score']
            },
            'model_saved': True,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get current alerts based on risk level and anomalies"""
    alerts = list(current_alerts)
    
    # Fallback evaluation using latest features
    if not alerts and risk_predictor and risk_predictor.is_trained:
        feature_vector = analyzer.get_feature_vector()
        if len(feature_vector) > 0:
            risk_level, confidence = risk_predictor.predict(feature_vector)
            
            if risk_level == 'high' and confidence > 0.7:
                alerts.append({
                    'type': 'high_risk',
                    'message': f'High crowd risk detected (confidence: {confidence:.2f})',
                    'severity': 'critical',
                    'timestamp': datetime.now().isoformat()
                })
            elif risk_level == 'medium' and confidence > 0.6:
                alerts.append({
                    'type': 'medium_risk',
                    'message': f'Medium crowd risk detected (confidence: {confidence:.2f})',
                    'severity': 'warning',
                    'timestamp': datetime.now().isoformat()
                })
    
    if not alerts and anomaly_detector and anomaly_detector.is_trained:
        feature_vector = analyzer.get_feature_vector()
        if len(feature_vector) > 0:
            is_anomaly, anomaly_score, details = anomaly_detector.detect_anomaly(feature_vector)
            
            if is_anomaly:
                alerts.append({
                    'type': 'anomaly',
                    'message': f'Anomalous crowd behavior detected (score: {anomaly_score:.2f})',
                    'severity': 'warning',
                    'timestamp': datetime.now().isoformat()
                })

    # Add capacity and rising trend alerts even when model predictions are unavailable
    if current_features:
        current_count = current_features.get('current_count', current_features.get('people_count', 0))
        growth_rate = current_features.get('growth_rate', 0.0)
        density_trend = current_features.get('density_trend', 0)
        avg_count = current_features.get('avg_count', current_count)
        threshold = int(30 * 0.8)
        existing_alert_types = {alert.get('type') for alert in alerts}

        if current_count >= threshold and 'capacity_threshold' not in existing_alert_types:
            alerts.append({
                'type': 'capacity_threshold',
                'message': f'Crowd reached {(current_count / 30) * 100:.1f}% of capacity ({current_count}/30). Take immediate action.',
                'severity': 'critical',
                'timestamp': datetime.now().isoformat()
            })
        elif (
            growth_rate >= 0.5 and
            density_trend == 1 and
            current_count >= avg_count * 1.1 and
            'rising_trend' not in existing_alert_types
        ):
            alerts.append({
                'type': 'rising_trend',
                'message': 'Crowd is gradually increasing. Please take necessary actions.',
                'severity': 'warning',
                'timestamp': datetime.now().isoformat()
            })
    
    return jsonify({
        'alerts': alerts,
        'alert_count': len(alerts),
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Initialize system
    initialize_system()
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
