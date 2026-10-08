"""
People Detection Module using YOLOv8
Handles video processing, people detection, and counting
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
from typing import List, Tuple, Dict
import logging

class PeopleDetector:
    """
    A class for detecting people in video frames using YOLOv8
    """
    
    def __init__(self, model_path: str = "yolov8n.pt", confidence_threshold: float = 0.15,
                 imgsz: int = 1280):
        """
        Initialize the PeopleDetector
        
        Args:
            model_path: Path to YOLOv8 model file
            confidence_threshold: Minimum confidence for detections.
                People in the Mall dataset are small and partly occluded, so
                0.5 misses most of them. 0.15 gave the closest counts to the
                ground truth in testing.
            imgsz: Inference resolution. Upscaling 640x480 frames to 1280
                lets YOLO see the small, distant people.
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.imgsz = imgsz
        # Optional trained CountCorrector (see src/count_corrector.py).
        # When set, the returned count is the calibrated one, while the
        # returned boxes are still the ones above confidence_threshold.
        self.count_corrector = None
        self.last_raw_count = 0
        self.person_class_id = 0  # COCO dataset class ID for 'person'
        self.frame_count = 0
        self.detection_history = []
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def detect_people(self, frame: np.ndarray) -> Tuple[List[Dict], int]:
        """
        Detect people in a single frame
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            Tuple of (detections, count)
            detections: List of dictionaries containing bbox, confidence, etc.
            count: Number of people detected
        """
        # Run YOLOv8 inference
        use_corrector = self.count_corrector is not None and self.count_corrector.is_trained
        run_conf = min(self.confidence_threshold, 0.05) if use_corrector else self.confidence_threshold
        results = self.model(
            frame,
            conf=run_conf,
            imgsz=self.imgsz,
            classes=[self.person_class_id],
            verbose=False,
        )
        
        detections = []
        people_count = 0
        all_boxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Check if detection is a person
                    if int(box.cls) == self.person_class_id:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        all_boxes.append([x1, y1, x2, y2, confidence])
                        if confidence < self.confidence_threshold:
                            continue
                        
                        detection = {
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': confidence,
                            'center': [int((x1 + x2) / 2), int((y1 + y2) / 2)],
                            'area': int((x2 - x1) * (y2 - y1))
                        }
                        
                        detections.append(detection)
                        people_count += 1
        
        self.last_raw_count = people_count
        if use_corrector:
            h, w = frame.shape[:2]
            people_count = self.count_corrector.predict(np.array(all_boxes).reshape(-1, 5), h, w)
        
        return detections, people_count
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       count: int) -> np.ndarray:
        """
        Draw bounding boxes and count on frame
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            count: Number of people detected
            
        Returns:
            Frame with drawn detections
        """
        # Create a copy to avoid modifying original
        annotated_frame = frame.copy()
        
        # Draw bounding boxes
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Person: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw count information
        count_text = f"People Count: {count}"
        cv2.putText(annotated_frame, count_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw frame number
        frame_text = f"Frame: {self.frame_count}"
        cv2.putText(annotated_frame, frame_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return annotated_frame
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_live: bool = True) -> Dict:
        """
        Process entire video and detect people in each frame
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
            show_live: Whether to display live processing
            
        Returns:
            Dictionary containing processing results
        """
        # Open video file
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        frame_data = []
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect people
            detections, count = self.detect_people(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, detections, count)
            
            # Store frame data
            frame_info = {
                'frame_number': self.frame_count,
                'timestamp': self.frame_count / fps,
                'people_count': count,
                'detections': detections
            }
            frame_data.append(frame_info)
            
            # Write frame if writer is available
            if writer:
                writer.write(annotated_frame)
            
            # Show live feed
            if show_live:
                cv2.imshow('Crowd Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Log progress
            if self.frame_count % 100 == 0:
                self.logger.info(f"Processed {self.frame_count}/{total_frames} frames")
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        processing_time = time.time() - start_time
        
        # Calculate statistics
        counts = [frame['people_count'] for frame in frame_data]
        stats = {
            'total_frames': self.frame_count,
            'processing_time': processing_time,
            'fps_processed': self.frame_count / processing_time,
            'avg_people_count': np.mean(counts),
            'max_people_count': np.max(counts),
            'min_people_count': np.min(counts),
            'total_detections': sum(counts),
            'frame_data': frame_data
        }
        
        self.logger.info(f"Processing completed in {processing_time:.2f} seconds")
        self.logger.info(f"Average people per frame: {stats['avg_people_count']:.2f}")
        
        return stats
    
    def process_webcam(self, camera_id: int = 0) -> None:
        """
        Process live webcam feed for real-time detection
        
        Args:
            camera_id: Camera device ID
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        self.logger.info("Starting webcam processing. Press 'q' to quit.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.frame_count += 1
            
            # Detect people
            detections, count = self.detect_people(frame)
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, detections, count)
            
            # Display
            cv2.imshow('Live Crowd Detection', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage
    detector = PeopleDetector()
    
    # Process video file (uncomment and provide path)
    # results = detector.process_video("data/sample_video.mp4", "data/output_video.mp4")
    # print(f"Detection results: {results}")
    
    # Process webcam (uncomment to use)
    # detector.process_webcam()
