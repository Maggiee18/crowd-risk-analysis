"""
Utility functions for the Crowd Risk Detection System
"""

import numpy as np
import cv2
import os
import json
import pickle
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def ensure_dir(directory: str) -> None:
    """Create directory if it doesn't exist"""
    if directory:
        os.makedirs(directory, exist_ok=True)

def _json_default(o):
    """Convert numpy types (int64, float32, arrays, bools) for json.dump"""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'Object of type {o.__class__.__name__} is not JSON serializable')

def save_json(data: Dict, file_path: str) -> None:
    """Save dictionary to JSON file"""
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=_json_default)

def load_json(file_path: str) -> Dict:
    """Load dictionary from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_model(model: Any, file_path: str) -> None:
    """Save machine learning model using pickle"""
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_model(file_path: str) -> Any:
    """Load machine learning model from pickle file"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def calculate_density_map(detections: List[Dict], frame_shape: Tuple[int, int], 
                        grid_size: int = 20) -> np.ndarray:
    """
    Calculate density map from people detections
    
    Args:
        detections: List of detection dictionaries
        frame_shape: (height, width) of the frame
        grid_size: Size of grid cells for density calculation
        
    Returns:
        Density map as numpy array
    """
    height, width = frame_shape
    grid_h = height // grid_size
    grid_w = width // grid_size
    
    density_map = np.zeros((grid_h, grid_w))
    
    for detection in detections:
        center_x, center_y = detection['center']
        grid_x = min(center_x // grid_size, grid_w - 1)
        grid_y = min(center_y // grid_size, grid_h - 1)
        
        if 0 <= grid_x < grid_w and 0 <= grid_y < grid_h:
            density_map[grid_y, grid_x] += 1
    
    return density_map

def smooth_data(data: List[float], window_size: int = 5) -> List[float]:
    """
    Apply moving average smoothing to data
    
    Args:
        data: List of numerical values
        window_size: Size of moving window
        
    Returns:
        Smoothed data
    """
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(data), i + window_size // 2 + 1)
        smoothed.append(np.mean(data[start_idx:end_idx]))
    
    return smoothed

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def plot_time_series(data: List[float], title: str, xlabel: str = "Time", 
                    ylabel: str = "Value", save_path: str = None) -> None:
    """
    Plot time series data
    
    Args:
        data: List of values to plot
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(12, 6))
    plt.plot(data, linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_heatmap(data: np.ndarray, title: str, save_path: str = None) -> None:
    """
    Create heatmap visualization
    
    Args:
        data: 2D numpy array for heatmap
        title: Plot title
        save_path: Path to save plot (optional)
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='YlOrRd', fmt='.1f')
    plt.title(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def calculate_optical_flow(prev_frame: np.ndarray, curr_frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate dense optical flow between consecutive frames using Farneback method.
    
    Args:
        prev_frame: Previous frame (BGR or grayscale)
        curr_frame: Current frame (BGR or grayscale)
        
    Returns:
        Tuple of (flow_magnitude, flow_angle)
    """
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
    
    # Calculate dense optical flow using Farneback method
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2,
        flags=0
    )
    
    # Calculate magnitude and angle from flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    
    return magnitude, angle

def normalize_features(features: np.ndarray) -> np.ndarray:
    """
    Normalize features to [0, 1] range
    
    Args:
        features: Input features array
        
    Returns:
        Normalized features
    """
    min_val = np.min(features, axis=0)
    max_val = np.max(features, axis=0)
    
    # Avoid division by zero
    range_val = max_val - min_val
    range_val[range_val == 0] = 1
    
    normalized = (features - min_val) / range_val
    return normalized

def create_video_writer(output_path: str, fps: int, frame_size: Tuple[int, int]) -> cv2.VideoWriter:
    """
    Create video writer object
    
    Args:
        output_path: Output video file path
        fps: Frames per second
        frame_size: (width, height) tuple
        
    Returns:
        VideoWriter object
    """
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)

def draw_text_with_background(frame: np.ndarray, text: str, position: Tuple[int, int], 
                             font_scale: float = 1.0, color: Tuple[int, int, int] = (255, 255, 255),
                             background_color: Tuple[int, int, int] = (0, 0, 0), thickness: int = 2) -> np.ndarray:
    """
    Draw text with background rectangle on frame
    
    Args:
        frame: Input frame
        text: Text to draw
        position: (x, y) position for text
        font_scale: Font scale factor
        color: Text color (B, G, R)
        background_color: Background color (B, G, R)
        thickness: Text thickness
        
    Returns:
        Frame with drawn text
    """
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate background rectangle coordinates
    x, y = position
    cv2.rectangle(frame, (x, y - text_height - baseline), 
                 (x + text_width, y + baseline), background_color, -1)
    
    # Draw text
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    
    return frame


def blur_faces(frame: np.ndarray, scale_factor: float = 1.1,
               min_neighbors: int = 5, blur_strength: int = 51) -> np.ndarray:
    """
    Detect and blur faces in a frame for privacy protection.
    Uses OpenCV Haar cascade classifier.
    
    Args:
        frame: Input BGR frame
        scale_factor: Scale factor for face detection cascade
        min_neighbors: Minimum neighbors for detection filtering
        blur_strength: Gaussian blur kernel size (must be odd)
        
    Returns:
        Frame with faces blurred
    """
    result = frame.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Load Haar cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor,
        minNeighbors=min_neighbors, minSize=(30, 30)
    )
    
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    for (x, y, w, h) in faces:
        # Extract face region and apply heavy Gaussian blur
        face_roi = result[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (blur_strength, blur_strength), 30)
        result[y:y+h, x:x+w] = blurred_face
    
    return result


def generate_zone_grid(frame_shape: Tuple[int, int], rows: int = 3, cols: int = 3) -> List[Dict]:
    """
    Divide frame into zones for zone-based analysis.
    
    Args:
        frame_shape: (height, width) of the frame
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        
    Returns:
        List of zone dictionaries with coordinates and IDs
    """
    height, width = frame_shape
    zone_h = height // rows
    zone_w = width // cols
    zones = []
    
    for r in range(rows):
        for c in range(cols):
            zone = {
                'id': f'zone_{r}_{c}',
                'label': f'Zone {r * cols + c + 1}',
                'x1': c * zone_w,
                'y1': r * zone_h,
                'x2': (c + 1) * zone_w,
                'y2': (r + 1) * zone_h,
                'center': [(c * zone_w + (c + 1) * zone_w) // 2,
                           (r * zone_h + (r + 1) * zone_h) // 2]
            }
            zones.append(zone)
    
    return zones
