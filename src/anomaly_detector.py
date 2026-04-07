"""
Anomaly Detection Module
Implements anomaly detection using Isolation Forest and other methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
import joblib
import logging
import matplotlib.pyplot as plt
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """
    Detects anomalies in crowd behavior patterns
    """
    
    def __init__(self, method: str = 'isolation_forest', contamination: float = 0.1):
        """
        Initialize AnomalyDetector
        
        Args:
            method: Anomaly detection method ('isolation_forest', 'one_class_svm', 'lof')
            contamination: Expected proportion of anomalies
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
        # History for real-time anomaly detection
        self.anomaly_history = deque(maxlen=100)
        self.anomaly_scores = deque(maxlen=100)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the anomaly detection model"""
        if self.method == 'isolation_forest':
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            )
        elif self.method == 'one_class_svm':
            self.model = OneClassSVM(
                nu=self.contamination,
                kernel='rbf',
                gamma='scale'
            )
        elif self.method == 'lof':
            self.model = LocalOutlierFactor(
                contamination=self.contamination,
                novelty=True,
                n_neighbors=20
            )
        else:
            raise ValueError(f"Unknown anomaly detection method: {self.method}")
    
    def train(self, features_df: pd.DataFrame) -> Dict:
        """
        Train the anomaly detection model
        
        Args:
            features_df: DataFrame containing normal behavior features
            
        Returns:
            Dictionary containing training results
        """
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features].fillna(0)
        self.feature_names = list(numeric_features)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if self.method == 'lof':
            # LOF needs to be fitted on normal data only
            self.model.fit(X_scaled)
        else:
            self.model.fit(X_scaled)
        
        self.is_trained = True
        
        # Get training predictions
        train_predictions = self.model.predict(X_scaled)
        train_scores = self.model.decision_function(X_scaled) if hasattr(self.model, 'decision_function') else None
        
        # Calculate training statistics
        n_anomalies = np.sum(train_predictions == -1)
        anomaly_rate = n_anomalies / len(train_predictions)
        
        self.logger.info(f"Model trained with {len(X)} samples")
        self.logger.info(f"Training anomaly rate: {anomaly_rate:.3f}")
        
        return {
            'n_samples': len(X),
            'n_anomalies': n_anomalies,
            'anomaly_rate': anomaly_rate,
            'train_predictions': train_predictions,
            'train_scores': train_scores
        }
    
    def detect_anomaly(self, features: np.ndarray) -> Tuple[bool, float, Dict]:
        """
        Detect if current features are anomalous
        
        Args:
            features: Feature array for current frame
            
        Returns:
            Tuple of (is_anomaly, anomaly_score, details)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before detection")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        is_anomaly = prediction == -1
        
        # Get anomaly score
        if hasattr(self.model, 'decision_function'):
            anomaly_score = self.model.decision_function(features_scaled)[0]
        elif hasattr(self.model, 'score_samples'):
            anomaly_score = self.model.score_samples(features_scaled)[0]
        else:
            anomaly_score = 0.0
        
        # Normalize score to [0, 1] where higher means more anomalous
        if self.method == 'isolation_forest':
            # Isolation Forest: lower scores are more anomalous
            normalized_score = max(0, -anomaly_score)
        else:
            # Other methods: lower scores are more anomalous
            normalized_score = max(0, -anomaly_score)
        
        # Update history
        self.anomaly_history.append(is_anomaly)
        self.anomaly_scores.append(normalized_score)
        
        # Calculate additional metrics
        recent_anomaly_rate = np.mean(list(self.anomaly_history)[-10:]) if len(self.anomaly_history) >= 10 else 0
        avg_anomaly_score = np.mean(list(self.anomaly_scores)[-10:]) if len(self.anomaly_scores) >= 10 else 0
        
        details = {
            'raw_score': anomaly_score,
            'normalized_score': normalized_score,
            'recent_anomaly_rate': recent_anomaly_rate,
            'avg_anomaly_score': avg_anomaly_score,
            'method': self.method
        }
        
        return is_anomaly, normalized_score, details
    
    def detect_sudden_surge(self, current_count: int, history_window: int = 10) -> Dict:
        """
        Detect sudden crowd surges using statistical methods
        
        Args:
            current_count: Current people count
            history_window: Window size for historical comparison
            
        Returns:
            Dictionary containing surge detection results
        """
        if len(self.anomaly_history) < history_window:
            return {
                'is_surge': False,
                'surge_magnitude': 0.0,
                'z_score': 0.0,
                'percentile': 50.0
            }
        
        # Get historical counts (simulated - in real use, this would come from actual data)
        # For now, we'll use the anomaly scores as a proxy
        historical_scores = list(self.anomaly_scores)[-history_window:]
        
        if len(historical_scores) == 0:
            return {
                'is_surge': False,
                'surge_magnitude': 0.0,
                'z_score': 0.0,
                'percentile': 50.0
            }
        
        # Calculate statistics
        mean_val = np.mean(historical_scores)
        std_val = np.std(historical_scores)
        
        # Z-score
        z_score = (current_count - mean_val) / std_val if std_val > 0 else 0
        
        # Percentile
        percentile = (np.sum(historical_scores <= current_count) / len(historical_scores)) * 100
        
        # Surge detection criteria
        is_surge = (z_score > 2.0) or (percentile > 90)
        surge_magnitude = max(0, z_score) if is_surge else 0
        
        return {
            'is_surge': is_surge,
            'surge_magnitude': surge_magnitude,
            'z_score': z_score,
            'percentile': percentile,
            'historical_mean': mean_val,
            'historical_std': std_val
        }
    
    def detect_abnormal_movement(self, movement_features: Dict) -> Dict:
        """
        Detect abnormal movement patterns
        
        Args:
            movement_features: Dictionary containing movement metrics
            
        Returns:
            Dictionary containing abnormal movement detection results
        """
        # Extract movement features
        avg_magnitude = movement_features.get('avg_magnitude', 0)
        max_magnitude = movement_features.get('max_magnitude', 0)
        movement_entropy = movement_features.get('movement_entropy', 0)
        angle_consistency = movement_features.get('angle_consistency', 0)
        movement_concentration = movement_features.get('movement_concentration', 0)
        
        # Define abnormal movement criteria
        abnormal_indicators = []
        
        # Very high movement speed
        if avg_magnitude > 50:
            abnormal_indicators.append('high_speed')
        
        # Very chaotic movement (high entropy, low consistency)
        if movement_entropy > 3.0 and angle_consistency < 0.3:
            abnormal_indicators.append('chaotic_movement')
        
        # Highly concentrated movement (potential stampede)
        if movement_concentration > 2.0 and max_magnitude > 100:
            abnormal_indicators.append('concentrated_movement')
        
        # Very dispersed movement (potential panic)
        if movement_concentration < 0.5 and avg_magnitude > 30:
            abnormal_indicators.append('dispersed_movement')
        
        # Calculate overall abnormality score
        abnormality_score = len(abnormal_indicators) / 4.0  # Normalize to [0, 1]
        
        return {
            'is_abnormal': len(abnormal_indicators) >= 2,
            'abnormality_score': abnormality_score,
            'indicators': abnormal_indicators,
            'features': {
                'avg_magnitude': avg_magnitude,
                'max_magnitude': max_magnitude,
                'movement_entropy': movement_entropy,
                'angle_consistency': angle_consistency,
                'movement_concentration': movement_concentration
            }
        }
    
    def get_anomaly_trend(self, window_size: int = 20) -> Dict:
        """
        Analyze anomaly trends over time
        
        Args:
            window_size: Size of the analysis window
            
        Returns:
            Dictionary containing trend analysis
        """
        if len(self.anomaly_scores) < window_size:
            return {
                'trend': 'insufficient_data',
                'trend_strength': 0.0,
                'recent_rate': 0.0,
                'overall_rate': 0.0
            }
        
        recent_scores = list(self.anomaly_scores)[-window_size:]
        recent_anomalies = list(self.anomaly_history)[-window_size:]
        
        # Calculate rates
        recent_rate = np.mean(recent_anomalies)
        overall_rate = np.mean(list(self.anomaly_history))
        
        # Calculate trend (simple linear regression on scores)
        x = np.arange(len(recent_scores))
        if len(x) > 1:
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 0.01:
                trend = 'increasing'
            elif slope < -0.01:
                trend = 'decreasing'
            else:
                trend = 'stable'
            
            # Normalize slope as trend strength
            trend_strength = min(abs(slope) * 10, 1.0)
        else:
            trend = 'stable'
            trend_strength = 0.0
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'recent_rate': recent_rate,
            'overall_rate': overall_rate,
            'avg_score': np.mean(recent_scores),
            'max_score': np.max(recent_scores)
        }
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'method': self.method,
            'contamination': self.contamination
        }
        
        joblib.dump(model_data, file_path)
        self.logger.info(f"Anomaly detection model saved to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """Load a trained model"""
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.method = model_data['method']
        self.contamination = model_data['contamination']
        self.is_trained = True
        
        self.logger.info(f"Anomaly detection model loaded from {file_path}")
    
    def plot_anomaly_scores(self, save_path: str = None) -> None:
        """Plot anomaly scores over time"""
        if len(self.anomaly_scores) == 0:
            self.logger.warning("No anomaly scores to plot")
            return
        
        plt.figure(figsize=(12, 6))
        scores = list(self.anomaly_scores)
        anomalies = list(self.anomaly_history)
        
        # Plot scores
        plt.plot(scores, label='Anomaly Score', color='blue', alpha=0.7)
        
        # Highlight anomalies
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                plt.scatter(i, scores[i], color='red', s=50, alpha=0.8)
        
        # Add threshold line
        threshold = np.percentile(scores, 90)
        plt.axhline(y=threshold, color='red', linestyle='--', label='Threshold')
        
        plt.title('Anomaly Detection Scores Over Time')
        plt.xlabel('Time (frames)')
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def reset_history(self) -> None:
        """Reset anomaly detection history"""
        self.anomaly_history.clear()
        self.anomaly_scores.clear()
        self.logger.info("Anomaly detection history reset")


class MultiMethodAnomalyDetector:
    """
    Combines multiple anomaly detection methods for robust detection
    """
    
    def __init__(self):
        self.detectors = {}
        self.weights = {}
    
    def add_detector(self, name: str, detector: AnomalyDetector, weight: float = 1.0):
        """Add an anomaly detector to the ensemble"""
        self.detectors[name] = detector
        self.weights[name] = weight
    
    def detect_ensemble(self, features: np.ndarray) -> Dict:
        """
        Make ensemble anomaly detection
        
        Returns:
            Dictionary containing ensemble results
        """
        individual_results = {}
        weighted_anomaly_score = 0
        total_weight = 0
        
        for name, detector in self.detectors.items():
            if detector.is_trained:
                is_anomaly, score, details = detector.detect_anomaly(features)
                individual_results[name] = {
                    'is_anomaly': is_anomaly,
                    'score': score,
                    'details': details
                }
                
                weighted_anomaly_score += score * self.weights[name]
                total_weight += self.weights[name]
        
        # Final decision
        final_score = weighted_anomaly_score / total_weight if total_weight > 0 else 0
        final_anomaly = final_score > 0.5
        
        # Voting mechanism
        anomaly_votes = sum(1 for result in individual_results.values() if result['is_anomaly'])
        consensus_anomaly = anomaly_votes > len(individual_results) / 2
        
        return {
            'final_anomaly': final_anomaly,
            'final_score': final_score,
            'consensus_anomaly': consensus_anomaly,
            'anomaly_votes': anomaly_votes,
            'total_detectors': len(individual_results),
            'individual_results': individual_results
        }


if __name__ == "__main__":
    # Example usage
    detector = AnomalyDetector(method='isolation_forest')
    
    # Create dummy normal data
    normal_data = pd.DataFrame({
        'current_count': np.random.normal(10, 3, 100),
        'density_per_frame': np.random.normal(0.5, 0.2, 100),
        'growth_rate': np.random.normal(0, 0.5, 100),
        'volatility': np.random.normal(0.5, 0.3, 100)
    })
    
    # Train detector
    training_results = detector.train(normal_data)
    print("Training results:", training_results)
    
    # Test with anomalous data
    anomalous_features = np.array([50, 2.0, 5.0, 2.0])  # High values
    is_anomaly, score, details = detector.detect_anomaly(anomalous_features)
    print(f"Anomaly detection: {is_anomaly}, Score: {score:.3f}")
