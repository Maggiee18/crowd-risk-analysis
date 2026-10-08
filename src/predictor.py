"""
Risk Prediction Module
Handles machine learning model training, prediction, and evaluation
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class RiskPredictor:
    """
    Machine learning model for crowd risk prediction
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize RiskPredictor
        
        Args:
            model_type: Type of model ('logistic_regression', 'random_forest', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.is_trained = False
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the ML model based on model_type"""
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        elif self.model_type == 'random_forest':
            self.model = RandomForestClassifier(random_state=42, n_estimators=100)
        elif self.model_type == 'svm':
            self.model = SVC(random_state=42, probability=True)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def generate_risk_labels(self, features_df: pd.DataFrame, 
                           method: str = 'threshold') -> pd.Series:
        """
        Generate risk labels based on features
        
        Args:
            features_df: DataFrame containing features
            method: Method for label generation ('threshold', 'percentile', 'clustering')
            
        Returns:
            Series of risk labels
        """
        if method == 'threshold':
            return self._generate_threshold_labels(features_df)
        elif method == 'percentile':
            return self._generate_percentile_labels(features_df)
        elif method == 'clustering':
            return self._generate_clustering_labels(features_df)
        else:
            raise ValueError(f"Unknown labeling method: {method}")
    
    def _generate_threshold_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate labels based on threshold rules"""
        labels = []
        
        for _, row in features_df.iterrows():
            count = row.get('current_count', 0)
            density = row.get('density_per_frame', 0)
            growth_rate = row.get('growth_rate', 0)
            volatility = row.get('volatility', 0)
            
            # Risk assessment rules
            if count < 5 and density < 0.1 and abs(growth_rate) < 0.5:
                risk = 'low'
            elif count < 15 and density < 0.5 and abs(growth_rate) < 2:
                risk = 'medium'
            else:
                risk = 'high'
            
            # Adjust based on volatility
            if volatility > 1.0 and risk != 'high':
                risk = 'medium' if risk == 'low' else 'high'
            
            labels.append(risk)
        
        return pd.Series(labels)
    
    def _generate_percentile_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate labels based on percentiles"""
        if 'current_count' not in features_df.columns:
            raise ValueError("DataFrame must contain 'current_count' column")
        
        counts = features_df['current_count']
        low_threshold = np.percentile(counts, 33)
        high_threshold = np.percentile(counts, 67)
        
        labels = []
        for count in counts:
            if count <= low_threshold:
                labels.append('low')
            elif count <= high_threshold:
                labels.append('medium')
            else:
                labels.append('high')
        
        return pd.Series(labels)
    
    def _generate_clustering_labels(self, features_df: pd.DataFrame) -> pd.Series:
        """Generate labels using K-means clustering"""
        from sklearn.cluster import KMeans
        
        # Select relevant features for clustering
        cluster_features = ['current_count', 'density_per_frame', 'growth_rate', 
                          'volatility', 'avg_magnitude']
        available_features = [f for f in cluster_features if f in features_df.columns]
        
        if len(available_features) < 2:
            raise ValueError("Not enough features for clustering")
        
        X = features_df[available_features].fillna(0)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Map clusters to risk levels based on centroid values
        centroids = kmeans.cluster_centers_
        count_idx = available_features.index('current_count') if 'current_count' in available_features else 0
        
        # Sort clusters by count values
        sorted_indices = np.argsort(centroids[:, count_idx])
        
        # Map to risk levels
        cluster_to_risk = {}
        for i, cluster_idx in enumerate(sorted_indices):
            if i == 0:
                cluster_to_risk[cluster_idx] = 'low'
            elif i == 1:
                cluster_to_risk[cluster_idx] = 'medium'
            else:
                cluster_to_risk[cluster_idx] = 'high'
        
        labels = [cluster_to_risk[label] for label in cluster_labels]
        return pd.Series(labels)
    
    def prepare_data(self, features_df: pd.DataFrame, labels: pd.Series = None,
                    test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training/testing
        
        Args:
            features_df: DataFrame containing features
            labels: Series of labels (if None, will generate)
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Generate labels if not provided
        if labels is None:
            labels = self.generate_risk_labels(features_df)
        
        # Select numeric features only
        numeric_features = features_df.select_dtypes(include=[np.number]).columns
        X = features_df[numeric_features].fillna(0)
        self.feature_names = list(numeric_features)
        
        # Encode labels
        y = self.label_encoder.fit_transform(labels)
        
        # Split data. Stratify only when every class has at least 2 samples,
        # otherwise sklearn raises "least populated class has only 1 member".
        class_counts = Counter(y)
        can_stratify = len(class_counts) > 1 and min(class_counts.values()) >= 2
        if not can_stratify:
            self.logger.warning(f"Not stratifying split, class counts: {dict(class_counts)}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42,
            stratify=y if can_stratify else None
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.logger.info(f"Data prepared: {X_train.shape[0]} training samples, {X_test.shape[0]} testing samples")
        self.logger.info(f"Class distribution: {Counter(y_train)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              hyperparameter_tuning: bool = False) -> Dict:
        """
        Train the ML model
        
        Args:
            X_train: Training features
            y_train: Training labels
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing training results
        """
        if hyperparameter_tuning:
            self.model = self._hyperparameter_tuning(X_train, y_train)
        else:
            self.model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train, y_train, cv=5, scoring='accuracy')
        
        self.is_trained = True
        self.logger.info(f"Model trained successfully. CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std()
        }
    
    def _hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray):
        """Perform hyperparameter tuning using GridSearchCV"""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
            }
        }
        
        param_grid = param_grids.get(self.model_type, {})
        
        if param_grid:
            grid_search = GridSearchCV(
                self.model, param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            self.logger.info(f"Best parameters: {grid_search.best_params_}")
            self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
            
            return grid_search.best_estimator_
        else:
            self.model.fit(X_train, y_train)
            return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test) if hasattr(self.model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Get class names. Pass explicit labels so the report still works when
        # the test split happens to be missing one of the classes.
        class_names = self.label_encoder.classes_
        all_labels = list(range(len(class_names)))
        
        # Classification report
        report = classification_report(
            y_test, y_pred, labels=all_labels, target_names=class_names,
            output_dict=True, zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred, labels=all_labels)
        
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': report,
            'confusion_matrix': cm,
            'class_names': class_names,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        self.logger.info(f"Evaluation completed - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
        
        return results
    
    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Make prediction on new data
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (predicted_label, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        # Ensure features are 2D
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        predicted_label = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get confidence
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = np.max(probabilities)
        else:
            confidence = 1.0
        
        return predicted_label, confidence
    
    def save_model(self, file_path: str) -> None:
        """Save the trained model"""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, file_path)
        self.logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: str) -> None:
        """Load a trained model"""
        model_data = joblib.load(file_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = True
        
        self.logger.info(f"Model loaded from {file_path}")
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                             save_path: str = None) -> None:
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, save_path: str = None) -> None:
        """Plot feature importance (for tree-based models)"""
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.title("Feature Importance")
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [self.feature_names[i] for i in indices], rotation=45)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.show()
        else:
            self.logger.warning("Feature importance not available for this model type")


class EnsemblePredictor:
    """
    Ensemble predictor combining multiple models
    """
    
    def __init__(self):
        self.predictors = {}
        self.weights = {}
    
    def add_predictor(self, name: str, predictor: RiskPredictor, weight: float = 1.0):
        """Add a predictor to the ensemble"""
        self.predictors[name] = predictor
        self.weights[name] = weight
    
    def predict(self, features: np.ndarray) -> Tuple[str, float, Dict]:
        """
        Make ensemble prediction
        
        Returns:
            Tuple of (predicted_label, confidence, individual_predictions)
        """
        individual_predictions = {}
        weighted_votes = {'low': 0, 'medium': 0, 'high': 0}
        total_confidence = 0
        
        for name, predictor in self.predictors.items():
            label, confidence = predictor.predict(features)
            individual_predictions[name] = {'label': label, 'confidence': confidence}
            
            weighted_votes[label] += confidence * self.weights[name]
            total_confidence += confidence * self.weights[name]
        
        # Final prediction based on weighted votes
        final_label = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_label] / total_confidence if total_confidence > 0 else 0
        
        return final_label, final_confidence, individual_predictions


if __name__ == "__main__":
    # Example usage
    predictor = RiskPredictor(model_type='random_forest')
    
    # Create dummy data
    dummy_features = pd.DataFrame({
        'current_count': np.random.randint(0, 50, 100),
        'density_per_frame': np.random.uniform(0, 2, 100),
        'growth_rate': np.random.uniform(-2, 2, 100),
        'volatility': np.random.uniform(0, 2, 100)
    })
    
    # Prepare and train
    X_train, X_test, y_train, y_test = predictor.prepare_data(dummy_features)
    training_results = predictor.train(X_train, y_train)
    
    # Evaluate
    evaluation_results = predictor.evaluate(X_test, y_test)
    print("Evaluation results:", evaluation_results)
