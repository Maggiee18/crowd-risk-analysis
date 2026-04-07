"""
Main Integration Script for Crowd Risk Detection System
Integrates all components and provides a unified interface
"""

import os
import sys
import cv2
import numpy as np
import argparse
import time
import threading
import json
from datetime import datetime
from typing import Dict, List, Optional

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import PeopleDetector
from analyzer import CrowdAnalyzer
from predictor import RiskPredictor, EnsemblePredictor
from anomaly_detector import AnomalyDetector, MultiMethodAnomalyDetector
from optimizer import RealTimeProcessor, HeatmapGenerator, AlertSystem, PerformanceMonitor
from utils import save_json, load_json, ensure_dir, get_timestamp

class CrowdRiskSystem:
    """
    Main system class that integrates all components
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the Crowd Risk Detection System
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.detector = None
        self.analyzer = None
        self.risk_predictor = None
        self.anomaly_detector = None
        self.ensemble_predictor = None
        self.realtime_processor = None
        self.heatmap_generator = None
        self.alert_system = None
        self.performance_monitor = None
        
        # System state
        self.is_initialized = False
        self.is_processing = False
        self.current_frame = None
        self.current_results = None
        
        # Initialize system
        self._initialize_system()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'detection': {
                'model_path': 'yolov8n.pt',
                'confidence_threshold': 0.5,
                'max_detections': 100
            },
            'analysis': {
                'frame_shape': [480, 640],
                'history_length': 30,
                'grid_size': 20
            },
            'prediction': {
                'model_type': 'random_forest',
                'ensemble': False
            },
            'anomaly_detection': {
                'method': 'isolation_forest',
                'contamination': 0.1,
                'multi_method': False
            },
            'optimization': {
                'enable_realtime': True,
                'max_workers': 4,
                'buffer_size': 30,
                'target_fps': 30.0
            },
            'alerts': {
                'risk_threshold': 0.7,
                'anomaly_threshold': 0.6,
                'enable_callbacks': True
            },
            'output': {
                'save_video': False,
                'save_features': True,
                'save_models': True,
                'output_dir': 'output'
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                config = {**default_config, **user_config}
                print(f"Configuration loaded from {config_path}")
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
                config = default_config
        else:
            config = default_config
            print("Using default configuration")
        
        return config
    
    def _initialize_system(self):
        """Initialize all system components"""
        try:
            print("Initializing Crowd Risk Detection System...")
            
            # Create output directory
            ensure_dir(self.config['output']['output_dir'])
            
            # Initialize detector
            print("Initializing people detector...")
            self.detector = PeopleDetector(
                model_path=self.config['detection']['model_path'],
                confidence_threshold=self.config['detection']['confidence_threshold']
            )
            
            # Initialize analyzer
            print("Initializing crowd analyzer...")
            frame_shape = tuple(self.config['analysis']['frame_shape'])
            self.analyzer = CrowdAnalyzer(
                frame_shape=frame_shape,
                history_length=self.config['analysis']['history_length']
            )
            
            # Initialize risk predictor
            print("Initializing risk predictor...")
            self.risk_predictor = RiskPredictor(
                model_type=self.config['prediction']['model_type']
            )
            
            # Initialize ensemble predictor if enabled
            if self.config['prediction']['ensemble']:
                self.ensemble_predictor = EnsemblePredictor()
                # Add multiple predictors
                self.ensemble_predictor.add_predictor('rf', 
                    RiskPredictor('random_forest'), weight=1.0)
                self.ensemble_predictor.add_predictor('svm', 
                    RiskPredictor('svm'), weight=0.8)
                self.ensemble_predictor.add_predictor('lr', 
                    RiskPredictor('logistic_regression'), weight=0.6)
            
            # Initialize anomaly detector
            print("Initializing anomaly detector...")
            if self.config['anomaly_detection']['multi_method']:
                self.anomaly_detector = MultiMethodAnomalyDetector()
                # Add multiple detectors
                self.anomaly_detector.add_detector('isolation_forest', 
                    AnomalyDetector('isolation_forest'), weight=1.0)
                self.anomaly_detector.add_detector('one_class_svm', 
                    AnomalyDetector('one_class_svm'), weight=0.8)
            else:
                self.anomaly_detector = AnomalyDetector(
                    method=self.config['anomaly_detection']['method'],
                    contamination=self.config['anomaly_detection']['contamination']
                )
            
            # Initialize optimization components
            if self.config['optimization']['enable_realtime']:
                print("Initializing real-time processor...")
                self.realtime_processor = RealTimeProcessor(
                    detector=self.detector,
                    analyzer=self.analyzer,
                    predictor=self.risk_predictor,
                    anomaly_detector=self.anomaly_detector,
                    max_workers=self.config['optimization']['max_workers'],
                    buffer_size=self.config['optimization']['buffer_size']
                )
            
            # Initialize heatmap generator
            self.heatmap_generator = HeatmapGenerator(
                grid_size=self.config['analysis']['grid_size']
            )
            
            # Initialize alert system
            self.alert_system = AlertSystem(
                risk_threshold=self.config['alerts']['risk_threshold'],
                anomaly_threshold=self.config['alerts']['anomaly_threshold']
            )
            
            # Initialize performance monitor
            self.performance_monitor = PerformanceMonitor()
            
            # Try to load pre-trained models
            self._load_pretrained_models()
            
            self.is_initialized = True
            print("System initialization completed successfully!")
            
        except Exception as e:
            print(f"Error initializing system: {e}")
            self.is_initialized = False
    
    def _load_pretrained_models(self):
        """Load pre-trained models if available"""
        models_dir = os.path.join(os.path.dirname(__file__), 'models')
        
        # Load risk prediction model
        risk_model_path = os.path.join(models_dir, 'risk_predictor.pkl')
        if os.path.exists(risk_model_path):
            try:
                self.risk_predictor.load_model(risk_model_path)
                print("Loaded pre-trained risk prediction model")
            except Exception as e:
                print(f"Error loading risk model: {e}")
        
        # Load anomaly detection model
        anomaly_model_path = os.path.join(models_dir, 'anomaly_detector.pkl')
        if os.path.exists(anomaly_model_path):
            try:
                self.anomaly_detector.load_model(anomaly_model_path)
                print("Loaded pre-trained anomaly detection model")
            except Exception as e:
                print(f"Error loading anomaly model: {e}")
    
    def process_video(self, video_path: str, output_path: str = None) -> Dict:
        """
        Process video file and generate comprehensive analysis
        
        Args:
            video_path: Path to input video
            output_path: Path for output video (optional)
            
        Returns:
            Dictionary containing processing results
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        print(f"Processing video: {video_path}")
        self.is_processing = True
        
        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Could not open video: {video_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            print(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
            
            # Setup video writer if needed
            writer = None
            if output_path or self.config['output']['save_video']:
                output_video_path = output_path or os.path.join(
                    self.config['output']['output_dir'], 
                    f"output_{get_timestamp()}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Process frames
            frame_results = []
            start_time = time.time()
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result = self._process_frame(frame, frame_count, fps)
                frame_results.append(result)
                
                # Draw annotations
                annotated_frame = self._draw_annotations(frame, result)
                
                # Add heatmap overlay
                if result['heatmap'] is not None:
                    heatmap_colored = self.heatmap_generator.apply_colormap(
                        result['heatmap'], (height, width)
                    )
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)
                
                # Write frame
                if writer:
                    writer.write(annotated_frame)
                
                # Progress update
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # Clean up
            cap.release()
            if writer:
                writer.release()
            
            processing_time = time.time() - start_time
            
            # Generate summary
            summary = self._generate_summary(frame_results, processing_time)
            
            # Save results
            if self.config['output']['save_features']:
                features_path = os.path.join(
                    self.config['output']['output_dir'], 
                    f"features_{get_timestamp()}.csv"
                )
                self.analyzer.save_features(features_path)
            
            summary_path = os.path.join(
                self.config['output']['output_dir'], 
                f"summary_{get_timestamp()}.json"
            )
            save_json(summary, summary_path)
            
            print(f"Video processing completed in {processing_time:.2f} seconds")
            print(f"Results saved to {self.config['output']['output_dir']}")
            
            return summary
            
        finally:
            self.is_processing = False
    
    def process_webcam(self, camera_id: int = 0, show_display: bool = True) -> None:
        """
        Process live webcam feed
        
        Args:
            camera_id: Camera device ID
            show_display: Whether to display live feed
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        print("Starting webcam processing. Press 'q' to quit.")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        self.is_processing = True
        frame_count = 0
        
        try:
            while self.is_processing:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result = self._process_frame(frame, frame_count)
                
                # Draw annotations
                annotated_frame = self._draw_annotations(frame, result)
                
                # Add heatmap overlay
                if result['heatmap'] is not None:
                    heatmap_colored = self.heatmap_generator.apply_colormap(
                        result['heatmap'], frame.shape[:2]
                    )
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, heatmap_colored, 0.3, 0)
                
                # Display
                if show_display:
                    cv2.imshow('Crowd Risk Detection', annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Update performance
                processing_time = result.get('processing_time', 0)
                fps = 1.0 / processing_time if processing_time > 0 else 0
                self.performance_monitor.update(processing_time, fps)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.is_processing = False
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, fps: float = None) -> Dict:
        """Process single frame and return results"""
        start_time = time.time()
        
        # Detect people
        detections, count = self.detector.detect_people(frame)
        
        # Analyze crowd
        features = self.analyzer.update_frame(frame, detections)
        
        # Generate heatmap
        heatmap = self.heatmap_generator.generate_heatmap(detections, frame.shape[:2])
        
        # Risk prediction
        risk_result = {}
        if self.risk_predictor.is_trained:
            feature_vector = self.analyzer.get_feature_vector()
            if len(feature_vector) > 0:
                if self.ensemble_predictor:
                    risk_level, confidence, individual_predictions = self.ensemble_predictor.predict(feature_vector)
                    risk_result = {
                        'risk_level': risk_level,
                        'confidence': confidence,
                        'individual_predictions': individual_predictions
                    }
                else:
                    risk_level, confidence = self.risk_predictor.predict(feature_vector)
                    risk_result = {
                        'risk_level': risk_level,
                        'confidence': confidence
                    }
        
        # Anomaly detection
        anomaly_result = {}
        if self.anomaly_detector.is_trained:
            feature_vector = self.analyzer.get_feature_vector()
            if len(feature_vector) > 0:
                if hasattr(self.anomaly_detector, 'detect_ensemble'):
                    anomaly_result = self.anomaly_detector.detect_ensemble(feature_vector)
                else:
                    is_anomaly, anomaly_score, details = self.anomaly_detector.detect_anomaly(feature_vector)
                    anomaly_result = {
                        'is_anomaly': is_anomaly,
                        'anomaly_score': anomaly_score,
                        'details': details
                    }
        
        # Generate alerts
        alerts = []
        if self.alert_system:
            alerts = self.alert_system.evaluate_risk(risk_result, anomaly_result, features)
        
        processing_time = time.time() - start_time
        
        result = {
            'frame_number': frame_number,
            'timestamp': frame_number / fps if fps else time.time(),
            'people_count': count,
            'detections': detections,
            'features': features,
            'heatmap': heatmap,
            'risk_prediction': risk_result,
            'anomaly_detection': anomaly_result,
            'alerts': alerts,
            'processing_time': processing_time
        }
        
        # Update current results
        self.current_results = result
        
        return result
    
    def _draw_annotations(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        # Draw detection boxes
        for detection in result['detections']:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw statistics
        y_offset = 30
        
        # People count
        count_text = f"People: {result['people_count']}"
        cv2.putText(annotated, count_text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        
        # Risk level
        if result['risk_prediction']:
            risk_level = result['risk_prediction'].get('risk_level', 'unknown')
            confidence = result['risk_prediction'].get('confidence', 0)
            risk_text = f"Risk: {risk_level.upper()} ({confidence:.2f})"
            
            # Color based on risk level
            color = {
                'low': (0, 255, 0),
                'medium': (0, 255, 255),
                'high': (0, 0, 255)
            }.get(risk_level, (255, 255, 255))
            
            cv2.putText(annotated, risk_text, (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            y_offset += 30
        
        # Anomaly indicator
        if result['anomaly_detection']:
            is_anomaly = result['anomaly_detection'].get('is_anomaly', False)
            if is_anomaly:
                anomaly_text = "ANOMALY DETECTED"
                cv2.putText(annotated, anomaly_text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                y_offset += 30
        
        # Frame info
        frame_text = f"Frame: {result['frame_number']}"
        cv2.putText(annotated, frame_text, (10, annotated.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Processing time
        proc_time = result.get('processing_time', 0)
        fps_text = f"FPS: {1.0/proc_time:.1f}" if proc_time > 0 else "FPS: N/A"
        cv2.putText(annotated, fps_text, (10, annotated.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated
    
    def _generate_summary(self, frame_results: List[Dict], processing_time: float) -> Dict:
        """Generate comprehensive summary of processing results"""
        if not frame_results:
            return {}
        
        # Extract metrics
        counts = [r['people_count'] for r in frame_results]
        processing_times = [r['processing_time'] for r in frame_results]
        
        # Risk distribution
        risk_counts = {'low': 0, 'medium': 0, 'high': 0}
        anomaly_count = 0
        alert_count = 0
        
        for result in frame_results:
            # Risk levels
            if result['risk_prediction']:
                risk_level = result['risk_prediction'].get('risk_level', 'low')
                risk_counts[risk_level] += 1
            
            # Anomalies
            if result['anomaly_detection']:
                if result['anomaly_detection'].get('is_anomaly', False):
                    anomaly_count += 1
            
            # Alerts
            alert_count += len(result.get('alerts', []))
        
        summary = {
            'processing_info': {
                'total_frames': len(frame_results),
                'processing_time_seconds': processing_time,
                'average_fps': len(frame_results) / processing_time,
                'average_processing_time': np.mean(processing_times)
            },
            'crowd_statistics': {
                'total_people_detected': sum(counts),
                'average_people_per_frame': np.mean(counts),
                'max_people_per_frame': np.max(counts),
                'min_people_per_frame': np.min(counts),
                'people_count_std': np.std(counts)
            },
            'risk_analysis': {
                'risk_distribution': risk_counts,
                'high_risk_percentage': (risk_counts['high'] / len(frame_results)) * 100,
                'medium_risk_percentage': (risk_counts['medium'] / len(frame_results)) * 100,
                'low_risk_percentage': (risk_counts['low'] / len(frame_results)) * 100
            },
            'anomaly_analysis': {
                'total_anomalies': anomaly_count,
                'anomaly_rate': (anomaly_count / len(frame_results)) * 100
            },
            'alert_analysis': {
                'total_alerts': alert_count,
                'alerts_per_frame': alert_count / len(frame_results)
            },
            'performance_metrics': self.performance_monitor.get_stats() if self.performance_monitor else {},
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def train_models(self, training_data_path: str) -> Dict:
        """
        Train ML models with provided data
        
        Args:
            training_data_path: Path to training data CSV
            
        Returns:
            Training results
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized")
        
        print(f"Training models with data from {training_data_path}")
        
        try:
            # Load training data
            import pandas as pd
            df = pd.read_csv(training_data_path)
            
            # Train risk predictor
            X_train, X_test, y_train, y_test = self.risk_predictor.prepare_data(df)
            training_results = self.risk_predictor.train(X_train, y_train)
            evaluation_results = self.risk_predictor.evaluate(X_test, y_test)
            
            # Train anomaly detector
            anomaly_training_results = self.anomaly_detector.train(df)
            
            # Save models
            if self.config['output']['save_models']:
                models_dir = os.path.join(os.path.dirname(__file__), 'models')
                ensure_dir(models_dir)
                
                self.risk_predictor.save_model(os.path.join(models_dir, 'risk_predictor.pkl'))
                self.anomaly_detector.save_model(os.path.join(models_dir, 'anomaly_detector.pkl'))
            
            print("Model training completed successfully!")
            
            return {
                'risk_predictor': {
                    'training_results': training_results,
                    'evaluation_results': evaluation_results
                },
                'anomaly_detector': anomaly_training_results
            }
            
        except Exception as e:
            print(f"Error training models: {e}")
            return {'error': str(e)}
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        return {
            'initialized': self.is_initialized,
            'processing': self.is_processing,
            'components': {
                'detector': self.detector is not None,
                'analyzer': self.analyzer is not None,
                'risk_predictor': self.risk_predictor.is_trained if self.risk_predictor else False,
                'anomaly_detector': self.anomaly_detector.is_trained if self.anomaly_detector else False,
                'realtime_processor': self.realtime_processor is not None,
                'alert_system': self.alert_system is not None
            },
            'performance': self.performance_monitor.get_stats() if self.performance_monitor else {},
            'current_results': self.current_results
        }


def main():
    """Main function for command line interface"""
    parser = argparse.ArgumentParser(description='Crowd Risk Detection System')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--webcam', action='store_true', help='Use webcam feed')
    parser.add_argument('--camera-id', type=int, default=0, help='Camera ID for webcam')
    parser.add_argument('--train', type=str, help='Path to training data')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Initialize system
    system = CrowdRiskSystem(args.config)
    
    if not system.is_initialized:
        print("Failed to initialize system. Exiting.")
        return
    
    # Update output directory if specified
    if args.output:
        system.config['output']['output_dir'] = args.output
    
    # Train models if requested
    if args.train:
        training_results = system.train_models(args.train)
        print("Training results:", training_results)
    
    # Process video if specified
    if args.video:
        results = system.process_video(args.video)
        print("Processing results:", results)
    
    # Process webcam if requested
    elif args.webcam:
        system.process_webcam(args.camera_id)
    
    # Show system status
    else:
        status = system.get_system_status()
        print("System Status:", json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
