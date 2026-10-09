"""
Performance Optimization Module
Handles real-time processing optimization, caching, and performance monitoring
"""

import numpy as np
import cv2
import time
import threading
import queue
from typing import Dict, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import psutil
import logging
from collections import deque
import multiprocessing as mp

class PerformanceMonitor:
    """Monitor system performance metrics"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.fps_history = deque(maxlen=window_size)
        self.cpu_history = deque(maxlen=window_size)
        self.memory_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.start_time = time.time()
        
    def update(self, processing_time: float, fps: float = None):
        """Update performance metrics"""
        self.processing_times.append(processing_time)
        if fps:
            self.fps_history.append(fps)
        
        # System metrics
        self.cpu_history.append(psutil.cpu_percent())
        self.memory_history.append(psutil.virtual_memory().percent)
    
    def get_stats(self) -> Dict:
        """Get current performance statistics"""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        return {
            'uptime_seconds': uptime,
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'current_fps': self.fps_history[-1] if self.fps_history else 0,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'current_processing_time': self.processing_times[-1] if self.processing_times else 0,
            'avg_cpu_usage': np.mean(self.cpu_history) if self.cpu_history else 0,
            'current_cpu_usage': self.cpu_history[-1] if self.cpu_history else 0,
            'avg_memory_usage': np.mean(self.memory_history) if self.memory_history else 0,
            'current_memory_usage': self.memory_history[-1] if self.memory_history else 0
        }

class FrameBuffer:
    """Thread-safe frame buffer for real-time processing"""
    
    def __init__(self, max_size: int = 30):
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
        self.not_empty = threading.Condition(self.lock)
        
    def put(self, frame: np.ndarray, timestamp: float = None):
        """Add frame to buffer"""
        with self.lock:
            if timestamp is None:
                timestamp = time.time()
            self.buffer.append((frame, timestamp))
            self.not_empty.notify()
    
    def get(self, timeout: float = None) -> Tuple[np.ndarray, float]:
        """Get frame from buffer"""
        with self.lock:
            while not self.buffer:
                if not self.not_empty.wait(timeout):
                    return None, None
            return self.buffer.popleft()
    
    def size(self) -> int:
        """Get buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        with self.lock:
            self.buffer.clear()

class RealTimeProcessor:
    """Optimized real-time processing pipeline"""
    
    def __init__(self, detector, analyzer, predictor=None, anomaly_detector=None,
                 max_workers: int = None, buffer_size: int = 30):
        self.detector = detector
        self.analyzer = analyzer
        self.predictor = predictor
        self.anomaly_detector = anomaly_detector
        
        # Performance optimization
        self.max_workers = max_workers or min(4, mp.cpu_count())
        self.frame_buffer = FrameBuffer(max_size=buffer_size)
        self.result_queue = queue.Queue(maxsize=buffer_size)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Processing state
        self.is_running = False
        self.processing_thread = None
        self.frame_counter = 0
        # Max seconds to wait for one frame (1280px CPU inference can take ~1s)
        self.frame_timeout = 30.0
        
        # Logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_processing(self):
        """Start real-time processing"""
        if self.is_running:
            return
        
        self.is_running = True
        # daemon=True so a crash in the caller can't leave the process hanging
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        self.logger.info("Real-time processing started")
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        self.executor.shutdown(wait=True)
        self.logger.info("Real-time processing stopped")
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to processing pipeline"""
        self.frame_buffer.put(frame)
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            # Get frame from buffer
            frame, timestamp = self.frame_buffer.get(timeout=0.1)
            if frame is None:
                continue
            
            start_time = time.time()
            
            try:
                # Process frame asynchronously
                future = self.executor.submit(self._process_frame, frame, timestamp)
                result = future.result(timeout=self.frame_timeout)
                
                self.frame_counter += 1
                result['frame_number'] = self.frame_counter
                result['processing_time'] = time.time() - start_time
                
                # Add result to queue
                if not self.result_queue.full():
                    self.result_queue.put(result)
                
                # Update performance metrics
                processing_time = time.time() - start_time
                fps = 1.0 / processing_time if processing_time > 0 else 0
                self.performance_monitor.update(processing_time, fps)
                
            except RuntimeError as e:
                # Executor was shut down (e.g. interpreter exiting): stop the loop
                self.logger.error(f"Stopping processing loop: {e}")
                self.is_running = False
            except Exception as e:
                self.logger.error(f"Error processing frame: {e}")
    
    def _process_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """Process single frame"""
        # Detect people
        detections, count = self.detector.detect_people(frame)
        
        # Analyze crowd
        features = self.analyzer.update_frame(frame, detections, people_count=count)
        
        # Risk prediction
        risk_result = {}
        if self.predictor and self.predictor.is_trained:
            feature_vector = self.analyzer.get_feature_vector(self.predictor.feature_names)
            if len(feature_vector) > 0:
                risk_level, confidence = self.predictor.predict(feature_vector)
                risk_result = {
                    'risk_level': risk_level,
                    'confidence': confidence
                }
        
        # Anomaly detection
        anomaly_result = {}
        if self.anomaly_detector and self.anomaly_detector.is_trained:
            feature_vector = self.analyzer.get_feature_vector(self.anomaly_detector.feature_names)
            if len(feature_vector) > 0:
                is_anomaly, anomaly_score, anomaly_details = self.anomaly_detector.detect_anomaly(feature_vector)
                anomaly_result = {
                    'is_anomaly': is_anomaly,
                    'anomaly_score': anomaly_score,
                    'details': anomaly_details
                }
        
        return {
            'timestamp': timestamp,
            'frame': frame,
            'detections': detections,
            'people_count': count,
            'features': features,
            'risk_prediction': risk_result,
            'anomaly_detection': anomaly_result
        }
    
    def get_latest_result(self) -> Optional[Dict]:
        """Get latest processing result"""
        try:
            return self.result_queue.get_nowait()
        except queue.Empty:
            return None
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        return self.performance_monitor.get_stats()

class CacheManager:
    """Intelligent caching system for frequently used data"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache = {}
        self.access_times = {}
        self.lock = threading.Lock()
    
    @lru_cache(maxsize=128)
    def get_cached_detection(self, frame_hash: int) -> Optional[List]:
        """Get cached detection results"""
        with self.lock:
            if frame_hash in self.cache:
                self.access_times[frame_hash] = time.time()
                return self.cache[frame_hash]
        return None
    
    def cache_detection(self, frame_hash: int, detections: List):
        """Cache detection results"""
        with self.lock:
            # Remove oldest items if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.access_times, key=self.access_times.get)
                del self.cache[oldest_key]
                del self.access_times[oldest_key]
            
            self.cache[frame_hash] = detections
            self.access_times[frame_hash] = time.time()
    
    def clear_cache(self):
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()

class AdaptiveQualityController:
    """Adaptive quality control for performance optimization"""
    
    def __init__(self, target_fps: float = 30.0):
        self.target_fps = target_fps
        self.current_quality = 1.0
        self.quality_history = deque(maxlen=10)
        self.fps_history = deque(maxlen=10)
        
    def update_quality(self, current_fps: float):
        """Adjust quality based on performance"""
        self.fps_history.append(current_fps)
        
        if len(self.fps_history) < 3:
            return
        
        avg_fps = np.mean(self.fps_history)
        
        # Adjust quality
        if avg_fps < self.target_fps * 0.8:
            # Reduce quality
            self.current_quality = max(0.3, self.current_quality * 0.9)
        elif avg_fps > self.target_fps * 1.2:
            # Increase quality
            self.current_quality = min(1.0, self.current_quality * 1.05)
        
        self.quality_history.append(self.current_quality)
    
    def get_frame_size(self, original_size: Tuple[int, int]) -> Tuple[int, int]:
        """Get adjusted frame size based on quality"""
        width, height = original_size
        scale_factor = 0.5 + (self.current_quality * 0.5)  # 0.5 to 1.0
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Ensure dimensions are divisible by 2 (for video codecs)
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        return new_width, new_height
    
    def get_detection_confidence(self) -> float:
        """Get adjusted detection confidence"""
        base_confidence = 0.5
        return base_confidence + (self.current_quality * 0.3)

class HeatmapGenerator:
    """Generate crowd density heatmaps"""
    
    def __init__(self, grid_size: int = 20):
        self.grid_size = grid_size
        self.heatmap_history = deque(maxlen=100)
        
    def generate_heatmap(self, detections: List[Dict], frame_shape: Tuple[int, int]) -> np.ndarray:
        """Generate density heatmap"""
        height, width = frame_shape[:2]
        grid_h = height // self.grid_size
        grid_w = width // self.grid_size
        
        # Initialize heatmap
        heatmap = np.zeros((grid_h, grid_w), dtype=np.float32)
        
        # Add detection contributions
        for detection in detections:
            center_x, center_y = detection['center']
            area = detection.get('area', 1000)
            
            # Calculate grid position
            grid_x = min(center_x // self.grid_size, grid_w - 1)
            grid_y = min(center_y // self.grid_size, grid_h - 1)
            
            # Add weighted contribution (larger area = more impact)
            weight = np.log(area + 1) / 10  # Normalize weight
            heatmap[grid_y, grid_x] += weight
        
        # Apply Gaussian smoothing for better visualization
        heatmap = cv2.GaussianBlur(heatmap, (5, 5), 0)
        
        # Normalize to [0, 255]
        if np.max(heatmap) > 0:
            heatmap = (heatmap / np.max(heatmap) * 255).astype(np.uint8)
        
        # Store in history
        self.heatmap_history.append(heatmap)
        
        return heatmap
    
    def get_accumulated_heatmap(self) -> np.ndarray:
        """Get accumulated heatmap over time"""
        if not self.heatmap_history:
            return np.zeros((20, 20), dtype=np.uint8)
        
        # Average all heatmaps in history
        accumulated = np.mean(list(self.heatmap_history), axis=0)
        
        # Normalize to [0, 255]
        if np.max(accumulated) > 0:
            accumulated = (accumulated / np.max(accumulated) * 255).astype(np.uint8)
        
        return accumulated
    
    def apply_colormap(self, heatmap: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Apply colormap to heatmap and resize to frame dimensions"""
        # Ensure heatmap is uint8 for colormap
        if heatmap.dtype != np.uint8:
            heatmap = np.clip(heatmap, 0, 255).astype(np.uint8)
        
        # Apply colormap
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Resize to frame dimensions
        height, width = frame_shape[:2]
        resized_heatmap = cv2.resize(colored_heatmap, (width, height))
        
        return resized_heatmap

class AlertSystem:
    """Intelligent alert system for high-risk situations"""
    
    def __init__(
        self,
        risk_threshold: float = 0.7,
        anomaly_threshold: float = 0.6,
        capacity_limit: int = 30,
        alert_ratio: float = 0.8,
        trend_window: int = 6,
        trend_min_slope: float = 0.5
    ):
        self.risk_threshold = risk_threshold
        self.anomaly_threshold = anomaly_threshold
        self.capacity_limit = capacity_limit
        self.alert_ratio = alert_ratio
        self.trend_window = trend_window
        self.trend_min_slope = trend_min_slope
        self.alert_history = deque(maxlen=50)
        self.active_alerts = {}
        self.alert_callbacks = []
        
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def evaluate_risk(self, risk_result: Dict, anomaly_result: Dict, 
                     features: Dict) -> List[Dict]:
        """Evaluate risk and generate alerts"""
        alerts = []
        current_time = time.time()
        
        # Check risk level
        if risk_result:
            risk_level = risk_result.get('risk_level', 'low')
            confidence = risk_result.get('confidence', 0.0)
            
            if risk_level == 'high' and confidence > self.risk_threshold:
                alert = {
                    'type': 'high_risk',
                    'severity': 'critical',
                    'message': f'High crowd risk detected (confidence: {confidence:.2f})',
                    'timestamp': current_time,
                    'data': risk_result
                }
                alerts.append(alert)
        
        # Check anomalies
        if anomaly_result:
            is_anomaly = anomaly_result.get('is_anomaly', False)
            anomaly_score = anomaly_result.get('anomaly_score', 0.0)
            
            if is_anomaly and anomaly_score > self.anomaly_threshold:
                alert = {
                    'type': 'anomaly',
                    'severity': 'warning',
                    'message': f'Anomalous behavior detected (score: {anomaly_score:.2f})',
                    'timestamp': current_time,
                    'data': anomaly_result
                }
                alerts.append(alert)
        
        # Panic: sudden fast, scattered (or rushing) movement, see panic_detector.py
        panic = (features or {}).get('panic') or {}
        if panic.get('is_panic'):
            if panic.get('kind') == 'scatter':
                msg = 'Panic detected: people are suddenly running in all directions. Act immediately.'
            else:
                msg = 'Sudden rush detected: the crowd is suddenly running the same way. Act immediately.'
            alerts.append({
                'type': 'panic',
                'severity': 'critical',
                'message': msg,
                'timestamp': current_time,
                'data': {k: panic.get(k) for k in
                         ('kind', 'fast_fraction', 'speed_ratio', 'coherence', 'tracked')}
            })

        # Check for sudden changes
        if features:
            # 'people_count' is this frame's count. 'current_count' comes from the
            # history *before* this frame was added, so it lags one frame behind
            # (and is 0 for the first 3 frames).
            count = features.get('people_count', features.get('current_count', 0))
            growth_rate = features.get('growth_rate', 0)
            volatility = features.get('volatility', 0)
            avg_count = features.get('avg_count', 0)
            density_trend = features.get('density_trend', 0)
            
            # Sudden crowd surge
            if growth_rate > 5.0:
                alert = {
                    'type': 'crowd_surge',
                    'severity': 'warning',
                    'message': f'Sudden crowd surge detected (growth rate: {growth_rate:.2f})',
                    'timestamp': current_time,
                    'data': {'growth_rate': growth_rate, 'count': count}
                }
                alerts.append(alert)
            
            # Capacity-based alerts (Mall dataset requirement)
            # WARNING at >= 80% of capacity, CRITICAL once capacity is exceeded.
            if self.capacity_limit > 0:
                utilization = (count / self.capacity_limit) * 100
                if count > self.capacity_limit:
                    alerts.append({
                        'type': 'capacity_exceeded',
                        'severity': 'critical',
                        'message': (
                            f'Capacity exceeded: {count}/{self.capacity_limit} people '
                            f'({utilization:.1f}%). Take immediate action.'
                        ),
                        'timestamp': current_time,
                        'data': {
                            'count': count,
                            'capacity_limit': self.capacity_limit,
                            'capacity_utilization_percent': utilization
                        }
                    })
                elif count >= self.capacity_limit * self.alert_ratio:
                    alerts.append({
                        'type': 'capacity_threshold',
                        'severity': 'warning',
                        'message': (
                            f'Crowd reached {utilization:.1f}% of capacity '
                            f'({count}/{self.capacity_limit}). Prepare crowd control.'
                        ),
                        'timestamp': current_time,
                        'data': {
                            'count': count,
                            'capacity_limit': self.capacity_limit,
                            'capacity_utilization_percent': utilization
                        }
                    })

            # Gradually rising crowd warning
            if (
                growth_rate >= self.trend_min_slope and
                density_trend == 1 and
                avg_count > 0 and
                count >= avg_count * 1.1
            ):
                alert = {
                    'type': 'rising_trend',
                    'severity': 'warning',
                    'message': (
                        'Crowd is gradually increasing. '
                        'Please take necessary actions.'
                    ),
                    'timestamp': current_time,
                    'data': {
                        'count': count,
                        'avg_count': avg_count,
                        'growth_rate': growth_rate,
                        'density_trend': density_trend
                    }
                }
                alerts.append(alert)
            
            # High volatility
            if volatility > 2.0:
                alert = {
                    'type': 'high_volatility',
                    'severity': 'warning',
                    'message': f'High crowd volatility detected (volatility: {volatility:.2f})',
                    'timestamp': current_time,
                    'data': {'volatility': volatility, 'count': count}
                }
                alerts.append(alert)
        
        # Process alerts
        for alert in alerts:
            self._process_alert(alert)
        
        return alerts
    
    def _process_alert(self, alert: Dict):
        """Process and store alert"""
        alert_id = f"{alert['type']}_{alert['timestamp']}"
        
        # Add to history
        self.alert_history.append(alert)
        
        # Add to active alerts
        self.active_alerts[alert_id] = alert
        
        # Trigger callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Error in alert callback: {e}")
        
        # Remove old alerts (older than 5 minutes)
        current_time = time.time()
        expired_alerts = [
            alert_id for alert_id, alert_data in self.active_alerts.items()
            if current_time - alert_data['timestamp'] > 300
        ]
        for alert_id in expired_alerts:
            del self.active_alerts[alert_id]
    
    def get_active_alerts(self) -> List[Dict]:
        """Get currently active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 20) -> List[Dict]:
        """Get recent alert history"""
        return list(self.alert_history)[-limit:]


if __name__ == "__main__":
    # Example usage
    print("Performance optimization modules loaded successfully")
