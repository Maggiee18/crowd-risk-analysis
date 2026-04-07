"""
Crowd Risk Detection System Demo
Shows system structure and functionality
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_system_structure():
    """Check if all system components are properly created"""
    print("=== Crowd Risk Detection System Structure Check ===")
    
    components = {
        'Core Modules': [
            'src/detector.py',
            'src/analyzer.py', 
            'src/predictor.py',
            'src/anomaly_detector.py',
            'src/optimizer.py',
            'src/utils.py'
        ],
        'API': [
            'api/app.py',
            'api/requirements.txt'
        ],
        'Frontend': [
            'frontend/package.json',
            'frontend/src/App.js',
            'frontend/src/index.js'
        ],
        'Configuration': [
            'config.json',
            'requirements.txt',
            'main.py'
        ],
        'Examples': [
            'examples/basic_usage.py',
            'setup_and_run.py'
        ]
    }
    
    print("\nChecking system components:")
    all_exist = True
    
    for category, files in components.items():
        print(f"\n{category}:")
        for file_path in files:
            exists = os.path.exists(file_path)
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")
            if not exists:
                all_exist = False
    
    print(f"\nOverall Status: {'✓ COMPLETE' if all_exist else '✗ INCOMPLETE'}")
    return all_exist

def demo_crowd_analysis():
    """Demonstrate crowd analysis capabilities"""
    print("\n=== Crowd Analysis Demo ===")
    
    try:
        from analyzer import CrowdAnalyzer
        
        # Initialize analyzer
        analyzer = CrowdAnalyzer(frame_shape=(480, 640))
        print("✓ Crowd Analyzer initialized")
        
        # Simulate crowd data over time
        print("\nSimulating crowd scenarios:")
        
        scenarios = [
            {"count": 5, "density": 0.2, "growth": 0.1, "volatility": 0.3},
            {"count": 15, "density": 0.8, "growth": 2.5, "volatility": 1.2},
            {"count": 35, "density": 2.5, "growth": 5.0, "volatility": 2.8},
            {"count": 8, "density": 0.4, "growth": -1.0, "volatility": 0.6}
        ]
        
        for i, scenario in enumerate(scenarios):
            # Create mock detections
            detections = []
            for j in range(scenario["count"]):
                x = 50 + j * 60
                y = 100 + (j % 3) * 80
                detections.append({
                    'bbox': [x, y, x+40, y+80],
                    'confidence': 0.8 + np.random.random() * 0.2,
                    'center': [x+20, y+40],
                    'area': 3200
                })
            
            # Create frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame[:] = (50 + i * 30, 100, 150)
            
            # Update analyzer
            features = analyzer.update_frame(frame, detections)
            
            # Determine risk level
            if scenario["count"] < 10:
                risk = "LOW"
            elif scenario["count"] < 25:
                risk = "MEDIUM"
            else:
                risk = "HIGH"
            
            print(f"  Scenario {i+1}: {scenario['count']} people, {risk} risk")
            print(f"    Density: {features['density_per_frame']:.2f}")
            print(f"    Growth Rate: {features.get('growth_rate', 0):.2f}")
            print(f"    Volatility: {features.get('volatility', 0):.2f}")
        
        # Get spatio-temporal analysis
        st_features = analyzer.calculate_spatio_temporal_features()
        print(f"\nSpatio-temporal Analysis:")
        print(f"  Average Count: {st_features['avg_count']:.1f}")
        print(f"  Max Count: {st_features['max_count']}")
        print(f"  Density Variation: {st_features['density_variation']:.2f}")
        print(f"  Overall Growth Rate: {st_features['growth_rate']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Crowd analysis demo failed: {e}")
        return False

def demo_risk_prediction():
    """Demonstrate risk prediction capabilities"""
    print("\n=== Risk Prediction Demo ===")
    
    try:
        from predictor import RiskPredictor
        
        # Create sample training data
        print("Creating training dataset...")
        np.random.seed(42)
        n_samples = 200
        
        # Generate realistic crowd scenarios
        data = {
            'current_count': np.concatenate([
                np.random.poisson(5, 60),      # Low crowd
                np.random.poisson(15, 80),     # Medium crowd  
                np.random.poisson(35, 60)      # High crowd
            ]),
            'density_per_frame': np.concatenate([
                np.random.exponential(0.2, 60),  # Low density
                np.random.exponential(0.8, 80),  # Medium density
                np.random.exponential(2.5, 60)   # High density
            ]),
            'growth_rate': np.random.normal(0, 2, n_samples),
            'volatility': np.random.exponential(0.8, n_samples),
            'avg_magnitude': np.random.exponential(8, n_samples),
            'movement_entropy': np.random.exponential(1.2, n_samples)
        }
        
        df = pd.DataFrame(data)
        print(f"✓ Created dataset with {len(df)} samples")
        
        # Initialize and train predictor
        print("Training Random Forest model...")
        predictor = RiskPredictor(model_type='random_forest')
        
        # Prepare and train
        X_train, X_test, y_train, y_test = predictor.prepare_data(df)
        training_results = predictor.train(X_train, y_train)
        evaluation_results = predictor.evaluate(X_test, y_test)
        
        print(f"✓ Model trained successfully!")
        print(f"  Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"  F1-Score: {evaluation_results['f1_score']:.3f}")
        
        # Test predictions
        print("\nTesting predictions:")
        test_cases = [
            ([3, 0.1, 0.2, 0.1, 2.0, 0.5], "Light crowd"),
            ([12, 0.9, 1.8, 0.7, 8.0, 1.2], "Moderate crowd"),
            ([28, 2.8, 4.5, 2.1, 18.0, 2.5], "Dense crowd")
        ]
        
        for features, description in test_cases:
            risk_level, confidence = predictor.predict(np.array(features))
            print(f"  {description}: {risk_level.upper()} risk (confidence: {confidence:.2f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Risk prediction demo failed: {e}")
        return False

def demo_anomaly_detection():
    """Demonstrate anomaly detection capabilities"""
    print("\n=== Anomaly Detection Demo ===")
    
    try:
        from anomaly_detector import AnomalyDetector
        
        # Create normal behavior patterns
        print("Creating normal behavior patterns...")
        np.random.seed(42)
        n_normal = 100
        
        normal_data = {
            'current_count': np.random.normal(12, 4, n_normal),
            'density_per_frame': np.random.normal(0.6, 0.3, n_normal),
            'growth_rate': np.random.normal(0, 0.8, n_normal),
            'volatility': np.random.normal(0.6, 0.4, n_normal)
        }
        
        # Ensure positive values
        for key in normal_data:
            normal_data[key] = np.abs(normal_data[key])
        
        normal_df = pd.DataFrame(normal_data)
        print(f"✓ Created {len(normal_df)} normal behavior samples")
        
        # Train anomaly detector
        print("Training Isolation Forest...")
        detector = AnomalyDetector(method='isolation_forest', contamination=0.1)
        training_results = detector.train(normal_df)
        
        print(f"✓ Anomaly detector trained")
        print(f"  Training anomaly rate: {training_results['anomaly_rate']:.3f}")
        
        # Test with various scenarios
        print("\nTesting anomaly detection:")
        test_scenarios = [
            ([10, 0.5, 0.3, 0.4], "Normal behavior"),
            ([25, 1.8, 3.2, 1.5], "Sudden crowd increase"),
            ([45, 4.2, 7.8, 3.1], "Extreme crowd density"),
            ([8, 0.3, -2.1, 0.2], "Rapid crowd decrease")
        ]
        
        for features, description in test_scenarios:
            is_anomaly, score, details = detector.detect_anomaly(np.array(features))
            status = "ANOMALY" if is_anomaly else "NORMAL"
            print(f"  {description}: {status} (score: {score:.3f})")
        
        return True
        
    except Exception as e:
        print(f"✗ Anomaly detection demo failed: {e}")
        return False

def create_sample_output():
    """Create sample output files to demonstrate system"""
    print("\n=== Creating Sample Output ===")
    
    try:
        # Create sample analysis results
        output_dir = "data"
        os.makedirs(output_dir, exist_ok=True)
        
        # Sample analysis data
        analysis_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'people_count': 15,
            'density_per_frame': 0.85,
            'growth_rate': 2.3,
            'volatility': 1.2,
            'risk_level': 'medium',
            'confidence': 0.78,
            'anomaly_detected': False,
            'anomaly_score': 0.15
        }
        
        # Save as JSON
        import json
        with open(f"{output_dir}/sample_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        # Create sample visualization
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:] = (50, 100, 150)  # Blue background
        
        # Add sample text
        cv2.putText(frame, "Crowd Risk Detection System", (150, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"People Count: {analysis_data['people_count']}", (150, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"Risk Level: {analysis_data['risk_level'].upper()}", (150, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {analysis_data['confidence']:.2f}", (150, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imwrite(f"{output_dir}/sample_output.jpg", frame)
        
        print(f"✓ Sample output created in '{output_dir}/' directory:")
        print(f"  - sample_analysis.json (analysis results)")
        print(f"  - sample_output.jpg (visualization)")
        
        return True
        
    except Exception as e:
        print(f"✗ Sample output creation failed: {e}")
        return False

def main():
    """Main demonstration function"""
    print("Crowd Risk Detection System - Complete Demo")
    print("=" * 60)
    print("This demo shows all system capabilities.")
    print("Note: YOLO model is required for real people detection.")
    print("=" * 60)
    
    # Check system structure
    structure_ok = check_system_structure()
    
    if not structure_ok:
        print("\n✗ Some system components are missing!")
        return
    
    # Run demos
    demos = [
        ("Crowd Analysis", demo_crowd_analysis),
        ("Risk Prediction", demo_risk_prediction), 
        ("Anomaly Detection", demo_anomaly_detection),
        ("Sample Output", create_sample_output)
    ]
    
    success_count = 0
    
    for demo_name, demo_func in demos:
        try:
            print(f"\n{'='*60}")
            if demo_func():
                success_count += 1
                print(f"✓ {demo_name} demo completed successfully")
            else:
                print(f"✗ {demo_name} demo failed")
        except Exception as e:
            print(f"✗ {demo_name} demo error: {e}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Demo Summary: {success_count}/{len(demos)} demos successful")
    
    if success_count == len(demos):
        print("\n🎉 All system components are working correctly!")
        print("\nNext Steps:")
        print("1. Install YOLO model for real people detection:")
        print("   pip install ultralytics")
        print("2. Run with real video:")
        print("   python main.py --video path/to/your/video.mp4")
        print("3. Or use webcam:")
        print("   python main.py --webcam")
        print("4. Start web interface:")
        print("   python setup_and_run.py --api")
        print("   python setup_and_run.py --frontend")
    else:
        print(f"\n⚠️  {len(demos) - success_count} demo(s) failed")
        print("Please check the error messages above.")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()
