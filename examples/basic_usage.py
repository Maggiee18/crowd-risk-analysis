"""
Basic Usage Examples for Crowd Risk Detection System
"""

import os
import sys
import cv2
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from detector import PeopleDetector
from analyzer import CrowdAnalyzer
from predictor import RiskPredictor
from anomaly_detector import AnomalyDetector
from main import CrowdRiskSystem

def example_1_basic_detection():
    """Example 1: Basic people detection"""
    print("=== Example 1: Basic People Detection ===")
    
    # Initialize detector
    detector = PeopleDetector()
    
    # Create a test frame (in real use, load from camera/video)
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 150, 200)  # Blue background
    
    # Add some test objects (simulating people)
    cv2.rectangle(test_frame, (100, 100), (200, 300), (255, 255, 255), -1)
    cv2.rectangle(test_frame, (300, 150), (400, 350), (255, 255, 255), -1)
    
    # Detect people
    detections, count = detector.detect_people(test_frame)
    
    print(f"Detected {count} people")
    for i, detection in enumerate(detections):
        print(f"Person {i+1}: bbox={detection['bbox']}, confidence={detection['confidence']:.2f}")
    
    # Draw detections
    annotated = detector.draw_detections(test_frame, detections, count)
    cv2.imwrite("example1_output.jpg", annotated)
    print("Output saved to example1_output.jpg")

def example_2_crowd_analysis():
    """Example 2: Crowd analysis features"""
    print("\n=== Example 2: Crowd Analysis ===")
    
    # Initialize components
    detector = PeopleDetector()
    analyzer = CrowdAnalyzer(frame_shape=(480, 640))
    
    # Simulate multiple frames
    for frame_num in range(10):
        # Create test frame with varying crowd sizes
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 100, 150)
        
        # Simulate increasing crowd
        num_people = 5 + frame_num * 2
        detections = []
        for i in range(num_people):
            x = 50 + i * 50
            y = 100 + (i % 3) * 100
            detections.append({
                'bbox': [x, y, x+40, y+80],
                'confidence': 0.8 + np.random.random() * 0.2,
                'center': [x+20, y+40],
                'area': 3200
            })
        
        # Update analyzer
        features = analyzer.update_frame(frame, detections)
        
        print(f"Frame {frame_num}: {features['people_count']} people, "
              f"density={features['density_per_frame']:.2f}, "
              f"growth_rate={features.get('growth_rate', 0):.2f}")
    
    # Get spatio-temporal features
    st_features = analyzer.calculate_spatio_temporal_features()
    print(f"\nSpatio-temporal features:")
    for key, value in st_features.items():
        print(f"  {key}: {value:.3f}")

def example_3_risk_prediction():
    """Example 3: Risk prediction with ML models"""
    print("\n=== Example 3: Risk Prediction ===")
    
    # Create dummy training data
    import pandas as pd
    
    # Generate sample features
    np.random.seed(42)
    n_samples = 200
    
    features_data = {
        'current_count': np.random.poisson(10, n_samples),
        'density_per_frame': np.random.exponential(0.5, n_samples),
        'growth_rate': np.random.normal(0, 1, n_samples),
        'volatility': np.random.exponential(0.3, n_samples),
        'avg_magnitude': np.random.exponential(5, n_samples),
        'movement_entropy': np.random.exponential(1, n_samples)
    }
    
    df = pd.DataFrame(features_data)
    
    # Initialize and train predictor
    predictor = RiskPredictor(model_type='random_forest')
    
    # Prepare data and train
    X_train, X_test, y_train, y_test = predictor.prepare_data(df)
    training_results = predictor.train(X_train, y_train)
    evaluation_results = predictor.evaluate(X_test, y_test)
    
    print("Training completed!")
    print(f"CV Accuracy: {training_results['cv_mean']:.3f}")
    print(f"Test Accuracy: {evaluation_results['accuracy']:.3f}")
    print(f"Test F1-Score: {evaluation_results['f1_score']:.3f}")
    
    # Make predictions on new data
    test_features = np.array([15, 1.2, 2.5, 0.8, 10.0, 1.5])
    risk_level, confidence = predictor.predict(test_features)
    
    print(f"\nPrediction for test features:")
    print(f"Risk Level: {risk_level}")
    print(f"Confidence: {confidence:.3f}")

def example_4_anomaly_detection():
    """Example 4: Anomaly detection"""
    print("\n=== Example 4: Anomaly Detection ===")
    
    # Create normal training data
    np.random.seed(42)
    n_normal = 100
    
    normal_data = {
        'current_count': np.random.normal(10, 3, n_normal),
        'density_per_frame': np.random.normal(0.5, 0.2, n_normal),
        'growth_rate': np.random.normal(0, 0.5, n_normal),
        'volatility': np.random.normal(0.5, 0.3, n_normal)
    }
    
    normal_df = pd.DataFrame(normal_data)
    
    # Initialize and train anomaly detector
    detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
    training_results = detector.train(normal_df)
    
    print(f"Anomaly detector trained on {training_results['n_samples']} samples")
    print(f"Training anomaly rate: {training_results['anomaly_rate']:.3f}")
    
    # Test with normal data
    normal_features = np.array([10, 0.5, 0.1, 0.4])
    is_anomaly, score, details = detector.detect_anomaly(normal_features)
    print(f"\nNormal data - Anomaly: {is_anomaly}, Score: {score:.3f}")
    
    # Test with anomalous data
    anomalous_features = np.array([50, 3.0, 5.0, 2.5])
    is_anomaly, score, details = detector.detect_anomaly(anomalous_features)
    print(f"Anomalous data - Anomaly: {is_anomaly}, Score: {score:.3f}")

def example_5_complete_system():
    """Example 5: Complete system integration"""
    print("\n=== Example 5: Complete System ===")
    
    # Initialize complete system
    system = CrowdRiskSystem()
    
    if not system.is_initialized:
        print("Failed to initialize system")
        return
    
    # Get system status
    status = system.get_system_status()
    print("System Status:")
    print(f"  Initialized: {status['initialized']}")
    print(f"  Components: {list(status['components'].keys())}")
    
    # Process a few simulated frames
    print("\nProcessing simulated frames...")
    
    for i in range(5):
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50 + i*20, 100, 150)
        
        # Add simulated people
        num_people = 3 + i
        detections = []
        for j in range(num_people):
            x = 100 + j * 80
            y = 150 + (j % 2) * 100
            detections.append({
                'bbox': [x, y, x+40, y+80],
                'confidence': 0.8 + np.random.random() * 0.2,
                'center': [x+20, y+40],
                'area': 3200
            })
        
        # Process frame
        result = system._process_frame(frame, i+1)
        
        print(f"Frame {i+1}: {result['people_count']} people")
        if result['risk_prediction']:
            risk = result['risk_prediction']
            print(f"  Risk: {risk['risk_level']} (confidence: {risk['confidence']:.2f})")
        if result['anomaly_detection']:
            anomaly = result['anomaly_detection']
            if anomaly.get('is_anomaly'):
                print(f"  ANOMALY DETECTED (score: {anomaly['anomaly_score']:.2f})")
    
    print("\nComplete system example finished!")

def example_6_real_time_simulation():
    """Example 6: Real-time processing simulation"""
    print("\n=== Example 6: Real-time Processing ===")
    
    from optimizer import RealTimeProcessor, PerformanceMonitor
    
    # Initialize system components
    detector = PeopleDetector()
    analyzer = CrowdAnalyzer(frame_shape=(480, 640))
    predictor = RiskPredictor()
    
    # Create real-time processor
    processor = RealTimeProcessor(
        detector=detector,
        analyzer=analyzer,
        predictor=predictor,
        max_workers=2,
        buffer_size=10
    )
    
    # Start processing
    processor.start_processing()
    
    # Simulate adding frames
    print("Simulating real-time frame processing...")
    
    for i in range(20):
        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (100, 150, 200)
        
        # Add frame to buffer
        processor.add_frame(frame)
        
        # Get latest result
        result = processor.get_latest_result()
        if result:
            print(f"Processed frame {result['frame_number']}: "
                  f"{result['people_count']} people, "
                  f"processing time: {result['processing_time']:.3f}s")
        
        # Simulate frame rate
        import time
        time.sleep(0.1)
    
    # Get performance stats
    stats = processor.get_performance_stats()
    print(f"\nPerformance Statistics:")
    print(f"  Average FPS: {stats['avg_fps']:.1f}")
    print(f"  Average Processing Time: {stats['avg_processing_time']:.3f}s")
    print(f"  CPU Usage: {stats['avg_cpu_usage']:.1f}%")
    
    # Stop processing
    processor.stop_processing()
    print("Real-time processing stopped")

if __name__ == "__main__":
    print("Crowd Risk Detection System - Basic Usage Examples")
    print("=" * 50)
    
    # Run all examples
    try:
        example_1_basic_detection()
        example_2_crowd_analysis()
        example_3_risk_prediction()
        example_4_anomaly_detection()
        example_5_complete_system()
        example_6_real_time_simulation()
        
        print("\n" + "=" * 50)
        print("All examples completed successfully!")
        print("Check the generated output files for visual results.")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
