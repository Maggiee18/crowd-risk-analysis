"""
Test script for Phase 1: People Detection and Counting
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from detector import PeopleDetector
import cv2
import numpy as np

def test_detector():
    """Test the PeopleDetector class"""
    print("Initializing PeopleDetector...")
    detector = PeopleDetector()
    
    # Test with a sample frame
    print("Creating test frame...")
    test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    test_frame[:] = (100, 150, 200)  # Fill with blue color
    
    # Test detection
    print("Testing people detection...")
    detections, count = detector.detect_people(test_frame)
    print(f"Detected {count} people in test frame")
    print(f"Detections: {detections}")
    
    # Test drawing
    print("Testing detection visualization...")
    annotated_frame = detector.draw_detections(test_frame, detections, count)
    
    # Save test frame
    cv2.imwrite("data/test_output.jpg", annotated_frame)
    print("Test frame saved to data/test_output.jpg")
    
    print("Phase 1 test completed successfully!")

def test_webcam():
    """Test webcam functionality"""
    print("Testing webcam (press 'q' to quit)...")
    detector = PeopleDetector()
    detector.process_webcam()

if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Run tests
    test_detector()
    
    # Uncomment to test webcam
    # test_webcam()
