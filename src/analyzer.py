"""
Crowd Analysis Module
Handles crowd density calculation, spatio-temporal features, and optical flow analysis
"""

import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import pandas as pd
from collections import deque
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from utils import calculate_density_map, smooth_data, calculate_optical_flow

class CrowdAnalyzer:
    """
    Analyzes crowd behavior and extracts features for risk prediction
    """
    
    def __init__(self, frame_shape: Tuple[int, int], history_length: int = 30):
        """
        Initialize CrowdAnalyzer
        
        Args:
            frame_shape: (height, width) of video frames
            history_length: Number of frames to keep in history
        """
        self.frame_shape = frame_shape
        self.history_length = history_length
        
        # Data storage
        self.frame_history = deque(maxlen=history_length)
        self.density_history = deque(maxlen=history_length)
        self.count_history = deque(maxlen=history_length)
        self.optical_flow_history = deque(maxlen=history_length)
        
        # Feature storage
        self.features_df = pd.DataFrame()
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def calculate_crowd_density(self, detections: List[Dict]) -> Dict:
        """
        Calculate various crowd density metrics
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary containing density metrics
        """
        if not detections:
            return {
                'people_count': 0,
                'density_per_frame': 0.0,
                'area_coverage': 0.0,
                'avg_inter_person_distance': 0.0,
                'density_map': np.zeros((20, 20))
            }
        
        # Basic count
        people_count = len(detections)
        
        # Calculate density map
        density_map = calculate_density_map(detections, self.frame_shape)
        
        # Frame area in pixels
        frame_area = self.frame_shape[0] * self.frame_shape[1]
        
        # Density per frame (people per square meter, assuming 1 pixel = 0.01m²)
        pixel_to_meter_sq = 0.0001  # Approximate conversion
        frame_area_meters = frame_area * pixel_to_meter_sq
        density_per_frame = people_count / frame_area_meters if frame_area_meters > 0 else 0
        
        # Area coverage (total bounding box area / frame area)
        total_bbox_area = sum(d['area'] for d in detections)
        area_coverage = total_bbox_area / frame_area if frame_area > 0 else 0
        
        # Calculate average inter-person distance
        centers = [d['center'] for d in detections]
        avg_distance = self._calculate_avg_inter_distance(centers)
        
        return {
            'people_count': people_count,
            'density_per_frame': density_per_frame,
            'area_coverage': area_coverage,
            'avg_inter_person_distance': avg_distance,
            'density_map': density_map
        }
    
    def _calculate_avg_inter_distance(self, centers: List[List[int]]) -> float:
        """Calculate average distance between all pairs of people"""
        if len(centers) < 2:
            return 0.0
        
        distances = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                dist = np.sqrt((centers[i][0] - centers[j][0])**2 + 
                             (centers[i][1] - centers[j][1])**2)
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def calculate_spatio_temporal_features(self) -> Dict:
        """
        Calculate spatio-temporal features from historical data
        
        Returns:
            Dictionary containing spatio-temporal features
        """
        if len(self.count_history) < 3:
            return self._get_default_features()
        
        # Convert to lists for easier manipulation
        counts = list(self.count_history)
        densities = [d['density_per_frame'] for d in self.density_history]
        
        # Density variation (standard deviation)
        density_variation = np.std(densities) if densities else 0.0
        
        # Growth rate (slope of linear regression on counts)
        time_points = list(range(len(counts)))
        if len(counts) >= 2:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_points, counts)
            growth_rate = slope
        else:
            growth_rate = 0.0
        
        # Count change rate (current - previous)
        count_change_rate = counts[-1] - counts[-2] if len(counts) >= 2 else 0.0
        
        # Acceleration (change in growth rate)
        if len(counts) >= 3:
            recent_growth = (counts[-1] - counts[-3]) / 2
            earlier_growth = (counts[-3] - counts[-5]) / 2 if len(counts) >= 5 else 0
            acceleration = recent_growth - earlier_growth
        else:
            acceleration = 0.0
        
        # Density trend (increasing/decreasing/stable)
        if len(densities) >= 3:
            recent_avg = np.mean(densities[-3:])
            earlier_avg = np.mean(densities[-6:-3]) if len(densities) >= 6 else np.mean(densities[:-3])
            if recent_avg > earlier_avg * 1.1:
                density_trend = 1  # Increasing
            elif recent_avg < earlier_avg * 0.9:
                density_trend = -1  # Decreasing
            else:
                density_trend = 0  # Stable
        else:
            density_trend = 0
        
        # Volatility (coefficient of variation)
        volatility = np.std(counts) / np.mean(counts) if np.mean(counts) > 0 else 0.0
        
        return {
            'density_variation': density_variation,
            'growth_rate': growth_rate,
            'count_change_rate': count_change_rate,
            'acceleration': acceleration,
            'density_trend': density_trend,
            'volatility': volatility,
            'current_count': counts[-1],
            'avg_count': np.mean(counts),
            'max_count': np.max(counts),
            'min_count': np.min(counts)
        }
    
    def calculate_optical_flow_features(self, prev_frame: np.ndarray, 
                                       curr_frame: np.ndarray) -> Dict:
        """
        Calculate optical flow features for movement analysis
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            
        Returns:
            Dictionary containing optical flow features
        """
        try:
            # Calculate optical flow
            magnitude, angle = calculate_optical_flow(prev_frame, curr_frame)
            
            # Movement metrics
            avg_magnitude = np.mean(magnitude)
            max_magnitude = np.max(magnitude)
            std_magnitude = np.std(magnitude)
            
            # Movement irregularity (entropy of magnitude distribution)
            hist, _ = np.histogram(magnitude.flatten(), bins=50, density=True)
            hist = hist[hist > 0]  # Remove zero bins
            movement_entropy = -np.sum(hist * np.log(hist + 1e-10)) if len(hist) > 0 else 0
            
            # Direction consistency
            angle_consistency = self._calculate_angle_consistency(angle, magnitude)
            
            # Movement concentration (how focused the movement is)
            movement_concentration = self._calculate_movement_concentration(magnitude)
            
            return {
                'avg_magnitude': avg_magnitude,
                'max_magnitude': max_magnitude,
                'std_magnitude': std_magnitude,
                'movement_entropy': movement_entropy,
                'angle_consistency': angle_consistency,
                'movement_concentration': movement_concentration
            }
        
        except Exception as e:
            self.logger.warning(f"Error calculating optical flow: {e}")
            return self._get_default_optical_flow_features()
    
    def _calculate_angle_consistency(self, angle: np.ndarray, magnitude: np.ndarray) -> float:
        """Calculate how consistent the movement directions are"""
        # Only consider significant movements
        significant_mask = magnitude > np.percentile(magnitude, 75)
        if np.sum(significant_mask) == 0:
            return 0.0
        
        significant_angles = angle[significant_mask]
        
        # Convert to unit vectors and calculate variance
        x_components = np.cos(significant_angles)
        y_components = np.sin(significant_angles)
        
        mean_x = np.mean(x_components)
        mean_y = np.mean(y_components)
        
        # Consistency is the magnitude of the mean vector (0-1)
        consistency = np.sqrt(mean_x**2 + mean_y**2)
        return consistency
    
    def _calculate_movement_concentration(self, magnitude: np.ndarray) -> float:
        """Calculate how concentrated the movement is in certain areas"""
        # Create spatial bins
        h, w = magnitude.shape
        grid_h, grid_w = h // 10, w // 10
        
        # Sum movement in each grid cell
        grid_movement = np.zeros((grid_h, grid_w))
        for i in range(grid_h):
            for j in range(grid_w):
                h_start, h_end = i * 10, min((i + 1) * 10, h)
                w_start, w_end = j * 10, min((j + 1) * 10, w)
                grid_movement[i, j] = np.sum(magnitude[h_start:h_end, w_start:w_end])
        
        # Calculate concentration (how unevenly distributed the movement is)
        total_movement = np.sum(grid_movement)
        if total_movement == 0:
            return 0.0
        
        expected_movement = total_movement / (grid_h * grid_w)
        concentration = np.std(grid_movement) / expected_movement if expected_movement > 0 else 0
        
        return concentration
    
    def update_frame(self, frame: np.ndarray, detections: List[Dict],
                     people_count: Optional[int] = None) -> Dict:
        """
        Update analyzer with new frame data
        
        Args:
            frame: Current frame
            detections: People detections in current frame
            people_count: Optional calibrated count (e.g. from CountCorrector).
                Defaults to len(detections).
            
        Returns:
            Dictionary containing all current features
        """
        # Calculate current density metrics
        density_metrics = self.calculate_crowd_density(detections)
        if people_count is not None and people_count != density_metrics['people_count']:
            frame_area_m2 = self.frame_shape[0] * self.frame_shape[1] * 0.0001
            density_metrics['people_count'] = people_count
            density_metrics['density_per_frame'] = people_count / frame_area_m2 if frame_area_m2 > 0 else 0
        
        # Calculate optical flow if we have previous frame
        optical_flow_features = {}
        if len(self.frame_history) > 0:
            optical_flow_features = self.calculate_optical_flow_features(
                self.frame_history[-1], frame
            )
        
        # Update history first so trend features include the current frame
        self.frame_history.append(frame.copy())
        self.density_history.append(density_metrics)
        self.count_history.append(density_metrics['people_count'])
        if optical_flow_features:
            self.optical_flow_history.append(optical_flow_features)
        
        # Calculate spatio-temporal features (now includes this frame)
        spatio_temporal_features = self.calculate_spatio_temporal_features()
        
        # Combine all features
        all_features = {
            **density_metrics,
            **spatio_temporal_features,
            **optical_flow_features,
            'timestamp': len(self.frame_history)
        }
        
        # Remove non-serializable objects for DataFrame storage
        features_for_df = {k: v for k, v in all_features.items() 
                         if k not in ['density_map']}
        
        # Add to features DataFrame
        new_row = pd.DataFrame([features_for_df])
        self.features_df = pd.concat([self.features_df, new_row], ignore_index=True)
        
        return all_features
    
    def get_feature_vector(self, feature_names: Optional[List[str]] = None) -> np.ndarray:
        """
        Get current feature vector for ML prediction
        
        Args:
            feature_names: Columns in the exact order a trained model expects.
                Without this the column order depends on which features
                appeared first (optical flow only exists from frame 2), so
                always pass model.feature_names when a model is trained.
        
        Returns:
            Numpy array of features
        """
        if len(self.features_df) == 0:
            return np.array([])
        
        # Get the latest row
        if feature_names:
            latest_features = self.features_df.iloc[-1].reindex(feature_names).values.astype(float)
        else:
            latest_features = self.features_df.iloc[-1].values
        
        # Handle any NaN values
        latest_features = np.nan_to_num(latest_features, nan=0.0)
        
        return latest_features
    
    def get_historical_features(self, window_size: int = 10) -> np.ndarray:
        """
        Get historical features for time series analysis
        
        Args:
            window_size: Number of recent frames to include
            
        Returns:
            Numpy array of historical features
        """
        if len(self.features_df) < window_size:
            window_size = len(self.features_df)
        
        if window_size == 0:
            return np.array([])
        
        recent_features = self.features_df.tail(window_size).values
        recent_features = np.nan_to_num(recent_features, nan=0.0)
        
        return recent_features.flatten()
    
    def reset(self) -> None:
        """Reset analyzer state"""
        self.frame_history.clear()
        self.density_history.clear()
        self.count_history.clear()
        self.optical_flow_history.clear()
        self.features_df = pd.DataFrame()
    
    def save_features(self, file_path: str) -> None:
        """Save features to CSV file"""
        self.features_df.to_csv(file_path, index=False)
        self.logger.info(f"Features saved to {file_path}")
    
    def load_features(self, file_path: str) -> None:
        """Load features from CSV file"""
        self.features_df = pd.read_csv(file_path)
        self.logger.info(f"Features loaded from {file_path}")
    
    def _get_default_features(self) -> Dict:
        """Get default feature values when insufficient data"""
        return {
            'density_variation': 0.0,
            'growth_rate': 0.0,
            'count_change_rate': 0.0,
            'acceleration': 0.0,
            'density_trend': 0,
            'volatility': 0.0,
            'current_count': 0,
            'avg_count': 0.0,
            'max_count': 0,
            'min_count': 0
        }
    
    def _get_default_optical_flow_features(self) -> Dict:
        """Get default optical flow features when calculation fails"""
        return {
            'avg_magnitude': 0.0,
            'max_magnitude': 0.0,
            'std_magnitude': 0.0,
            'movement_entropy': 0.0,
            'angle_consistency': 0.0,
            'movement_concentration': 0.0
        }


if __name__ == "__main__":
    # Example usage
    analyzer = CrowdAnalyzer(frame_shape=(480, 640))
    
    # Test with dummy data
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    dummy_detections = [
        {'center': [100, 100], 'area': 1000},
        {'center': [200, 200], 'area': 1200}
    ]
    
    features = analyzer.update_frame(dummy_frame, dummy_detections)
    print("Features extracted:", features.keys())
