"""
LSTM-based Temporal Crowd Density Predictor
Uses PyTorch to predict future crowd density from historical time-series data.
Includes synthetic data generation for initial training.
"""

import numpy as np
import logging
import os
import json
from typing import List, Dict, Tuple, Optional
from collections import deque

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Try to import PyTorch; provide a numpy-only fallback if unavailable
# --------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — LSTM predictor will use linear fallback")


# ==========================================================================
# PyTorch LSTM Model
# ==========================================================================
if TORCH_AVAILABLE:

    class CrowdLSTM(nn.Module):
        """LSTM network for crowd density time-series prediction."""

        def __init__(self, input_size: int = 4, hidden_size: int = 64,
                     num_layers: int = 2, output_steps: int = 10,
                     dropout: float = 0.2):
            super().__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.output_steps = output_steps

            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0
            )
            self.fc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size // 2, output_steps)
            )

        def forward(self, x):
            # x shape: (batch, seq_len, input_size)
            lstm_out, _ = self.lstm(x)
            # Take the last time step
            last_hidden = lstm_out[:, -1, :]
            return self.fc(last_hidden)

    class CrowdDataset(Dataset):
        """PyTorch Dataset for crowd time-series windows."""

        def __init__(self, sequences: np.ndarray, targets: np.ndarray):
            self.sequences = torch.FloatTensor(sequences)
            self.targets = torch.FloatTensor(targets)

        def __len__(self):
            return len(self.sequences)

        def __getitem__(self, idx):
            return self.sequences[idx], self.targets[idx]


# ==========================================================================
# Synthetic Data Generator
# ==========================================================================

class SyntheticCrowdDataGenerator:
    """Generate realistic synthetic crowd time-series data for training."""

    @staticmethod
    def generate_scenario(n_steps: int = 500, scenario: str = 'normal') -> np.ndarray:
        """
        Generate a synthetic crowd metric time-series.

        Args:
            n_steps: Number of time steps
            scenario: One of 'normal', 'gradual_buildup', 'sudden_surge',
                      'event_dispersal', 'oscillating'

        Returns:
            Array of shape (n_steps, 4) with columns:
            [people_count, density, avg_speed, flow_magnitude]
        """
        t = np.arange(n_steps, dtype=np.float32)
        noise = np.random.normal(0, 0.5, n_steps)

        if scenario == 'normal':
            count = 10 + 3 * np.sin(t / 50) + noise
        elif scenario == 'gradual_buildup':
            count = 5 + t * 0.06 + 2 * np.sin(t / 30) + noise
        elif scenario == 'sudden_surge':
            count = np.where(t < n_steps * 0.6, 8 + noise,
                             8 + 20 * (1 - np.exp(-(t - n_steps * 0.6) / 20)) + noise)
        elif scenario == 'event_dispersal':
            count = np.where(t < n_steps * 0.4,
                             25 + 5 * np.sin(t / 20) + noise,
                             25 * np.exp(-(t - n_steps * 0.4) / 80) + noise)
        elif scenario == 'oscillating':
            count = 15 + 10 * np.sin(t / 40) + 5 * np.sin(t / 15) + noise
        else:
            count = 10 + noise

        count = np.clip(count, 0, None)
        density = count * 0.05 + np.random.normal(0, 0.02, n_steps)
        density = np.clip(density, 0, None)
        speed = 5 + 2 * np.random.randn(n_steps) + np.abs(np.diff(count, prepend=count[0]))
        speed = np.clip(speed, 0, None)
        flow_mag = speed * 0.3 + np.random.normal(0, 0.5, n_steps)
        flow_mag = np.clip(flow_mag, 0, None)

        data = np.stack([count, density, speed, flow_mag], axis=1).astype(np.float32)
        return data

    @staticmethod
    def generate_training_data(window_size: int = 30, prediction_steps: int = 10,
                               n_scenarios: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training windows from multiple scenarios.

        Returns:
            (X, y) where X shape is (N, window_size, 4) and y shape is (N, prediction_steps)
        """
        scenarios = ['normal', 'gradual_buildup', 'sudden_surge',
                     'event_dispersal', 'oscillating']
        all_X = []
        all_y = []

        for _ in range(n_scenarios):
            scenario = np.random.choice(scenarios)
            series = SyntheticCrowdDataGenerator.generate_scenario(500, scenario)
            # Sliding window
            for i in range(len(series) - window_size - prediction_steps):
                X_window = series[i:i + window_size]
                # Predict future counts (column 0)
                y_future = series[i + window_size:i + window_size + prediction_steps, 0]
                all_X.append(X_window)
                all_y.append(y_future)

        return np.array(all_X, dtype=np.float32), np.array(all_y, dtype=np.float32)


# ==========================================================================
# Main Predictor Class
# ==========================================================================

class LSTMCrowdPredictor:
    """
    LSTM-based crowd density predictor.
    Uses PyTorch when available, falls back to simple linear extrapolation.
    """

    def __init__(self, window_size: int = 30, prediction_steps: int = 10,
                 input_features: int = 4, hidden_size: int = 64,
                 num_layers: int = 2):
        self.window_size = window_size
        self.prediction_steps = prediction_steps
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.model = None
        self.is_trained = False
        self.device = 'cpu'
        self._history = deque(maxlen=window_size + 10)
        # residual=True: the network predicts the change from the last
        # observed count instead of the absolute count. On noisy real data
        # this beats predicting raw counts.
        self.residual = False

        # Normalisation stats
        self._mean = None
        self._std = None

        if TORCH_AVAILABLE:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = CrowdLSTM(
                input_size=input_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                output_steps=prediction_steps
            ).to(self.device)
            logger.info(f"LSTM model initialised on {self.device}")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def make_windows(self, series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Turn a real (T, 4) sequence of [count, density, avg_speed, flow_mag]
        into sliding (X, y) training windows.
        """
        series = np.asarray(series, dtype=np.float32)
        X, y = [], []
        for i in range(len(series) - self.window_size - self.prediction_steps + 1):
            X.append(series[i:i + self.window_size])
            y.append(series[i + self.window_size:i + self.window_size + self.prediction_steps, 0])
        return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

    def train_on_sequence(self, series: np.ndarray, epochs: int = 50,
                          batch_size: int = 64, lr: float = 0.001) -> Dict:
        """Train on a real recorded sequence (e.g. the Mall dataset)."""
        X, y = self.make_windows(series)
        return self.train(epochs=epochs, batch_size=batch_size, lr=lr, X=X, y=y)

    def train(self, epochs: int = 50, batch_size: int = 64,
              lr: float = 0.001, n_scenarios: int = 60,
              X: np.ndarray = None, y: np.ndarray = None) -> Dict:
        """
        Train the LSTM. Uses the given (X, y) windows, or synthetic crowd
        data when none are given.

        Returns:
            Dict with training metrics.
        """
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch not available — skipping LSTM training")
            self.is_trained = True
            return {'status': 'fallback_linear', 'epochs': 0}

        if X is None or y is None:
            logger.info("Generating synthetic training data …")
            X, y = SyntheticCrowdDataGenerator.generate_training_data(
                window_size=self.window_size,
                prediction_steps=self.prediction_steps,
                n_scenarios=n_scenarios
            )

        # Normalise
        self._mean = X.reshape(-1, self.input_features).mean(axis=0)
        self._std = X.reshape(-1, self.input_features).std(axis=0) + 1e-8
        X_norm = (X - self._mean) / self._std

        # Also normalise targets using count column stats
        y_std = self._std[0]
        if self.residual:
            y_norm = (y - X[:, -1, 0:1]) / y_std
        else:
            y_norm = (y - self._mean[0]) / y_std

        # Split 90/10
        split = int(len(X_norm) * 0.9)
        train_ds = CrowdDataset(X_norm[:split], y_norm[:split])
        val_ds = CrowdDataset(X_norm[split:], y_norm[split:])
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

        optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=5, factor=0.5
        )

        best_val_loss = float('inf')
        history = {'train_loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Train
            self.model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                pred = self.model(xb)
                loss = criterion(pred, yb)
                optimiser.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimiser.step()
                train_losses.append(loss.item())

            # Validate
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    pred = self.model(xb)
                    val_losses.append(criterion(pred, yb).item())

            avg_train = np.mean(train_losses)
            avg_val = np.mean(val_losses)
            history['train_loss'].append(avg_train)
            history['val_loss'].append(avg_val)
            scheduler.step(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} — "
                            f"train: {avg_train:.4f}  val: {avg_val:.4f}")

        self.is_trained = True
        logger.info(f"Training complete — best val loss: {best_val_loss:.4f}")

        return {
            'status': 'trained',
            'epochs': epochs,
            'best_val_loss': best_val_loss,
            'final_train_loss': history['train_loss'][-1],
            'final_val_loss': history['val_loss'][-1]
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def add_observation(self, people_count: float, density: float,
                        avg_speed: float, flow_magnitude: float):
        """Append a new time step observation to the rolling history."""
        self._history.append([people_count, density, avg_speed, flow_magnitude])

    def predict(self, input_sequence: np.ndarray = None) -> Dict:
        """
        Predict future crowd density.

        Args:
            input_sequence: Optional (window_size, input_features) array.
                            If None, uses internal history.

        Returns:
            Dict with 'predictions' (list of predicted counts),
            'confidence' and 'method'.
        """
        # Build input from history if not provided
        if input_sequence is None:
            if len(self._history) < self.window_size:
                return {
                    'predictions': [],
                    'confidence': 0.0,
                    'method': 'insufficient_data',
                    'steps_needed': self.window_size - len(self._history)
                }
            input_sequence = np.array(list(self._history)[-self.window_size:],
                                      dtype=np.float32)

        # ---- PyTorch LSTM path ----
        if TORCH_AVAILABLE and self.model is not None and self.is_trained and self._mean is not None:
            self.model.eval()
            x_norm = (input_sequence - self._mean) / self._std
            x_tensor = torch.FloatTensor(x_norm).unsqueeze(0).to(self.device)

            with torch.no_grad():
                pred_norm = self.model(x_tensor).cpu().numpy()[0]

            # De-normalise
            if self.residual:
                predictions = (pred_norm * self._std[0] + input_sequence[-1, 0]).tolist()
            else:
                predictions = (pred_norm * self._std[0] + self._mean[0]).tolist()
            predictions = [max(0, p) for p in predictions]

            return {
                'predictions': [round(p, 1) for p in predictions],
                'confidence': 0.85,
                'method': 'lstm'
            }

        # ---- Linear fallback ----
        counts = input_sequence[:, 0]
        slope = np.polyfit(np.arange(len(counts)), counts, 1)[0]
        last_val = counts[-1]
        predictions = [max(0, last_val + slope * (i + 1))
                       for i in range(self.prediction_steps)]
        return {
            'predictions': [round(p, 1) for p in predictions],
            'confidence': 0.4,
            'method': 'linear_extrapolation'
        }

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, directory: str):
        """Save model weights and normalisation stats."""
        os.makedirs(directory, exist_ok=True)
        meta = {
            'window_size': self.window_size,
            'prediction_steps': self.prediction_steps,
            'input_features': self.input_features,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'mean': self._mean.tolist() if self._mean is not None else None,
            'std': self._std.tolist() if self._std is not None else None,
            'residual': self.residual
        }
        with open(os.path.join(directory, 'lstm_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)

        if TORCH_AVAILABLE and self.model is not None:
            torch.save(self.model.state_dict(),
                       os.path.join(directory, 'lstm_weights.pt'))
        logger.info(f"LSTM predictor saved to {directory}")

    def load(self, directory: str):
        """Load model weights and normalisation stats."""
        meta_path = os.path.join(directory, 'lstm_meta.json')
        if not os.path.exists(meta_path):
            logger.warning(f"No LSTM model found in {directory}")
            return

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        self.window_size = meta['window_size']
        self.prediction_steps = meta['prediction_steps']
        self.input_features = meta['input_features']
        self.hidden_size = meta['hidden_size']
        self.num_layers = meta['num_layers']
        self.residual = meta.get('residual', False)

        if meta['mean'] is not None:
            self._mean = np.array(meta['mean'], dtype=np.float32)
            self._std = np.array(meta['std'], dtype=np.float32)

        if TORCH_AVAILABLE:
            self.model = CrowdLSTM(
                input_size=self.input_features,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                output_steps=self.prediction_steps
            ).to(self.device)

            weights_path = os.path.join(directory, 'lstm_weights.pt')
            if os.path.exists(weights_path):
                self.model.load_state_dict(
                    torch.load(weights_path, map_location=self.device)
                )
                self.model.eval()

        self.is_trained = True
        logger.info(f"LSTM predictor loaded from {directory}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    predictor = LSTMCrowdPredictor(window_size=30, prediction_steps=10)

    # Train on synthetic data
    result = predictor.train(epochs=30, n_scenarios=30)
    print(f"Training result: {result}")

    # Generate a test scenario and predict
    test_data = SyntheticCrowdDataGenerator.generate_scenario(100, 'gradual_buildup')
    for row in test_data[:40]:
        predictor.add_observation(*row)

    prediction = predictor.predict()
    print(f"Prediction: {prediction}")
