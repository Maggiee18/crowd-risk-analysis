"""
Setup and Run Script for Crowd Risk Detection System
Handles installation, setup, and execution of the complete system
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements_files = [
        "requirements.txt",
        "api/requirements.txt"
    ]
    
    for req_file in requirements_files:
        if os.path.exists(req_file):
            print(f"Installing from {req_file}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
    
    print("Requirements installation completed!")

def setup_directories():
    """Create necessary directories"""
    print("Setting up directories...")
    
    directories = [
        "models",
        "data",
        "output",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def download_models():
    """Download pre-trained models"""
    print("Downloading YOLOv8 model...")
    
    try:
        from ultralytics import YOLO
        
        # Download YOLOv8 model
        model = YOLO('yolov8n.pt')
        print("YOLOv8 model downloaded successfully!")
        
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("The model will be downloaded automatically on first use.")

def run_tests():
    """Run basic system tests"""
    print("Running system tests...")
    
    try:
        # Test basic imports
        import cv2
        import numpy as np
        from ultralytics import YOLO
        import sklearn
        
        print("✓ All required packages imported successfully")
        
        # Test YOLO model
        model = YOLO('yolov8n.pt')
        print("✓ YOLO model loaded successfully")
        
        # Test OpenCV
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imwrite("test_output.jpg", test_image)
        os.remove("test_output.jpg")
        print("✓ OpenCV functionality verified")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    
    return True

def run_example():
    """Run basic example"""
    print("Running basic example...")
    
    try:
        # Import and run basic example
        sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
        
        from detector import PeopleDetector
        import numpy as np
        import cv2
        
        # Create test frame
        detector = PeopleDetector()
        test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        test_frame[:] = (100, 150, 200)
        
        # Test detection
        detections, count = detector.detect_people(test_frame)
        print(f"✓ Detection test completed: {count} people detected")
        
        # Test drawing
        annotated = detector.draw_detections(test_frame, detections, count)
        cv2.imwrite("data/example_output.jpg", annotated)
        print("✓ Example output saved to data/example_output.jpg")
        
    except Exception as e:
        print(f"Example failed: {e}")
        return False
    
    return True

def start_api_server():
    """Start the Flask API server"""
    print("Starting API server...")
    
    try:
        os.chdir("api")
        subprocess.run([sys.executable, "app.py"])
    except KeyboardInterrupt:
        print("\nAPI server stopped")
    except Exception as e:
        print(f"Error starting API server: {e}")

def start_frontend():
    """Start the React frontend"""
    print("Starting React frontend...")
    
    try:
        os.chdir("frontend")
        
        # Check if node_modules exists
        if not os.path.exists("node_modules"):
            print("Installing frontend dependencies...")
            subprocess.run(["npm", "install"])
        
        # Start development server
        subprocess.run(["npm", "start"])
        
    except KeyboardInterrupt:
        print("\nFrontend stopped")
    except Exception as e:
        print(f"Error starting frontend: {e}")

def run_full_system():
    """Run the complete system with webcam"""
    print("Starting complete crowd risk detection system...")
    
    try:
        from main import CrowdRiskSystem
        
        # Initialize system
        system = CrowdRiskSystem("config.json")
        
        if not system.is_initialized:
            print("Failed to initialize system")
            return
        
        print("System initialized successfully!")
        print("Starting webcam processing...")
        print("Press 'q' to quit")
        
        # Start webcam processing
        system.process_webcam(show_display=True)
        
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error running system: {e}")

def create_sample_training_data():
    """Create sample training data for testing"""
    print("Creating sample training data...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'current_count': np.random.poisson(15, n_samples),
            'density_per_frame': np.random.exponential(0.8, n_samples),
            'growth_rate': np.random.normal(0, 1.5, n_samples),
            'volatility': np.random.exponential(0.5, n_samples),
            'avg_magnitude': np.random.exponential(8, n_samples),
            'movement_entropy': np.random.exponential(1.2, n_samples),
            'angle_consistency': np.random.uniform(0, 1, n_samples),
            'movement_concentration': np.random.exponential(1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Save to data directory
        df.to_csv("data/sample_training_data.csv", index=False)
        print("✓ Sample training data saved to data/sample_training_data.csv")
        
    except Exception as e:
        print(f"Error creating sample data: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Crowd Risk Detection System Setup')
    parser.add_argument('--install', action='store_true', help='Install requirements')
    parser.add_argument('--setup', action='store_true', help='Setup directories and models')
    parser.add_argument('--test', action='store_true', help='Run system tests')
    parser.add_argument('--example', action='store_true', help='Run basic example')
    parser.add_argument('--api', action='store_true', help='Start API server')
    parser.add_argument('--frontend', action='store_true', help='Start frontend')
    parser.add_argument('--run', action='store_true', help='Run complete system')
    parser.add_argument('--create-data', action='store_true', help='Create sample training data')
    parser.add_argument('--all', action='store_true', help='Run complete setup')
    
    args = parser.parse_args()
    
    if args.all or args.install:
        install_requirements()
    
    if args.all or args.setup:
        setup_directories()
        download_models()
    
    if args.all or args.test:
        if not run_tests():
            print("Tests failed. Please check installation.")
            return
    
    if args.all or args.example:
        run_example()
    
    if args.create_data:
        create_sample_training_data()
    
    if args.api:
        start_api_server()
    
    if args.frontend:
        start_frontend()
    
    if args.run:
        run_full_system()
    
    if not any(vars(args).values()):
        print("Crowd Risk Detection System Setup")
        print("Use --help to see available options")
        print("\nQuick start:")
        print("  python setup_and_run.py --all    # Complete setup")
        print("  python setup_and_run.py --run    # Run system")
        print("  python setup_and_run.py --api    # Start API only")
        print("  python setup_and_run.py --frontend # Start frontend only")

if __name__ == "__main__":
    main()
