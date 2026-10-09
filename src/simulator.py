"""
Simulation Module
Generates mock crowd scenarios for testing and demonstration
when no real video feed is available.
"""

import numpy as np
import cv2
import time
import math
import logging
from typing import Dict, List, Tuple, Optional, Generator
from collections import deque

logger = logging.getLogger(__name__)


# ==========================================================================
# Simulated Person
# ==========================================================================

class SimulatedPerson:
    """A single simulated person with position, velocity, and behaviour."""

    _next_id = 0

    def __init__(self, x: float, y: float, bounds: Tuple[int, int, int, int]):
        self.id = SimulatedPerson._next_id
        SimulatedPerson._next_id += 1
        self.x = x
        self.y = y
        self.vx = np.random.uniform(-2, 2)
        self.vy = np.random.uniform(-2, 2)
        self.bounds = bounds  # (x_min, y_min, x_max, y_max)
        self.alive = True
        self.width = np.random.randint(25, 45)
        self.height = np.random.randint(60, 100)

    def update(self, dt: float = 1.0, panic: bool = False):
        """Update position with simple physics."""
        if panic:
            # Panic mode: faster random movement
            self.vx += np.random.uniform(-5, 5)
            self.vy += np.random.uniform(-5, 5)
            speed_limit = 15.0
        else:
            # Random walk with momentum
            self.vx += np.random.uniform(-0.5, 0.5)
            self.vy += np.random.uniform(-0.5, 0.5)
            speed_limit = 4.0

        # Clamp speed
        speed = math.sqrt(self.vx**2 + self.vy**2)
        if speed > speed_limit:
            self.vx = (self.vx / speed) * speed_limit
            self.vy = (self.vy / speed) * speed_limit

        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off bounds
        x_min, y_min, x_max, y_max = self.bounds
        if self.x < x_min:
            self.x = x_min
            self.vx = abs(self.vx)
        elif self.x > x_max - self.width:
            self.x = x_max - self.width
            self.vx = -abs(self.vx)
        if self.y < y_min:
            self.y = y_min
            self.vy = abs(self.vy)
        elif self.y > y_max - self.height:
            self.y = y_max - self.height
            self.vy = -abs(self.vy)

    def get_detection(self) -> Dict:
        """Return detection dict compatible with the pipeline."""
        x1, y1 = int(self.x), int(self.y)
        x2, y2 = x1 + self.width, y1 + self.height
        return {
            'bbox': [x1, y1, x2, y2],
            'confidence': round(0.7 + np.random.random() * 0.3, 3),
            'center': [(x1 + x2) // 2, (y1 + y2) // 2],
            'area': self.width * self.height
        }


# ==========================================================================
# Scenario Definitions
# ==========================================================================

SCENARIOS = {
    'normal': {
        'description': 'Normal crowd flow — steady low-to-moderate density',
        'initial_people': 10,
        'spawn_rate': 0.05,
        'despawn_rate': 0.04,
        'max_people': 20,
        'panic_probability': 0.0,
        'duration_seconds': 30
    },
    'gradual_buildup': {
        'description': 'Gradually increasing crowd over time',
        'initial_people': 5,
        'spawn_rate': 0.15,
        'despawn_rate': 0.02,
        'max_people': 50,
        'panic_probability': 0.0,
        'duration_seconds': 45
    },
    'sudden_surge': {
        'description': 'Sudden crowd surge at a specific time',
        'initial_people': 10,
        'spawn_rate': 0.05,
        'despawn_rate': 0.03,
        'max_people': 60,
        'panic_probability': 0.0,
        'surge_at': 0.4,  # 40% into the scenario
        'surge_amount': 25,
        'duration_seconds': 40
    },
    'panic_event': {
        'description': 'Normal crowd that suddenly panics',
        'initial_people': 20,
        'spawn_rate': 0.03,
        'despawn_rate': 0.02,
        'max_people': 30,
        'panic_probability': 0.0,
        'panic_start': 0.4,
        'panic_end': 0.8,
        'duration_seconds': 30
    },
    'evacuation': {
        'description': 'Large crowd evacuating — rapid decrease',
        'initial_people': 40,
        'spawn_rate': 0.0,
        'despawn_rate': 0.2,
        'max_people': 40,
        'panic_probability': 0.3,
        'duration_seconds': 25
    },
    'concert': {
        'description': 'Concert-like event with waves of density',
        'initial_people': 15,
        'spawn_rate': 0.1,
        'despawn_rate': 0.06,
        'max_people': 45,
        'panic_probability': 0.01,
        'duration_seconds': 50
    }
}


# ==========================================================================
# Crowd Simulator
# ==========================================================================

class CrowdSimulator:
    """
    Simulates crowd scenarios with synthetic video frames and detection data.
    Can be used as a drop-in replacement for real video when testing.
    """

    def __init__(self, frame_width: int = 640, frame_height: int = 480,
                 fps: int = 15):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.fps = fps
        self.people: List[SimulatedPerson] = []
        self.bounds = (20, 20, frame_width - 20, frame_height - 20)
        self.frame_count = 0
        self.scenario_name = None
        self.scenario_config = None
        self.is_running = False

        # History for features
        self._count_history = deque(maxlen=100)

    def _spawn_person(self):
        """Spawn a new person at a random edge."""
        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
        if edge == 'top':
            x = np.random.uniform(self.bounds[0], self.bounds[2])
            y = self.bounds[1]
        elif edge == 'bottom':
            x = np.random.uniform(self.bounds[0], self.bounds[2])
            y = self.bounds[3] - 80
        elif edge == 'left':
            x = self.bounds[0]
            y = np.random.uniform(self.bounds[1], self.bounds[3])
        else:
            x = self.bounds[2] - 40
            y = np.random.uniform(self.bounds[1], self.bounds[3])

        person = SimulatedPerson(x, y, self.bounds)
        self.people.append(person)

    def start_scenario(self, scenario: str = 'normal') -> Dict:
        """
        Start a simulation scenario.

        Args:
            scenario: Name of the scenario from SCENARIOS dict

        Returns:
            Scenario configuration dict
        """
        if scenario not in SCENARIOS:
            logger.warning(f"Unknown scenario '{scenario}', using 'normal'")
            scenario = 'normal'

        self.scenario_name = scenario
        self.scenario_config = SCENARIOS[scenario].copy()
        self.people.clear()
        self.frame_count = 0
        SimulatedPerson._next_id = 0
        self.is_running = True

        # Spawn initial people
        for _ in range(self.scenario_config['initial_people']):
            x = np.random.uniform(self.bounds[0] + 50, self.bounds[2] - 50)
            y = np.random.uniform(self.bounds[1] + 50, self.bounds[3] - 50)
            person = SimulatedPerson(x, y, self.bounds)
            self.people.append(person)

        logger.info(f"Started scenario: {scenario} — "
                    f"{self.scenario_config['description']}")

        return self.scenario_config

    def step(self, steps: int = 1) -> Dict:
        """
        Advance simulation by one or more frames.

        Returns:
            Dict with frame data, detections, and features.
        """
        if not self.is_running:
            return {}

        config = self.scenario_config
        self.frame_count += steps
        total_frames = config['duration_seconds'] * self.fps
        progress = self.frame_count / total_frames

        # Check if scenario is complete
        if self.frame_count >= total_frames:
            self.is_running = False

        # Determine if currently in panic mode
        panic = False
        if 'panic_start' in config and 'panic_end' in config:
            if config['panic_start'] <= progress <= config['panic_end']:
                panic = True
        if np.random.random() < config['panic_probability']:
            panic = True

        # Sudden surge
        if 'surge_at' in config:
            surge_frame = int(config['surge_at'] * total_frames)
            # If we crossed the surge frame in this step
            if (self.frame_count - steps) < surge_frame <= self.frame_count:
                for _ in range(config['surge_amount']):
                    self._spawn_person()

        # Spawn / despawn
        for _ in range(steps):
            if len(self.people) < config['max_people']:
                if np.random.random() < config['spawn_rate']:
                    self._spawn_person()
            if len(self.people) > 0 and np.random.random() < config['despawn_rate']:
                self.people.pop(np.random.randint(len(self.people)))

        # Update all people. Velocities are in pixels per frame, so dt is the
        # number of frames advanced. (It used to be seconds, 1/15 per frame:
        # people moved ~0.2 px per frame, which int() rounding turned into
        # zero, so the crowd stood still and even "panic" barely moved.)
        dt = float(steps)
        for person in self.people:
            person.update(dt=dt, panic=panic)

        # Generate detections
        detections = [p.get_detection() for p in self.people]
        count = len(detections)
        self._count_history.append(count)

        # Render frame
        frame = self._render_frame(detections, panic)

        # Compute basic features
        frame_area = self.frame_width * self.frame_height
        density = count / (frame_area * 0.0001) if frame_area > 0 else 0

        avg_speed = 0.0
        if self.people:
            speeds = [math.sqrt(p.vx**2 + p.vy**2) for p in self.people]
            avg_speed = float(np.mean(speeds))

        growth_rate = 0.0
        if len(self._count_history) >= 5:
            recent = list(self._count_history)[-5:]
            growth_rate = (recent[-1] - recent[0]) / 5.0

        return {
            'frame': frame,
            'frame_number': self.frame_count,
            'timestamp': self.frame_count / self.fps,
            'detections': detections,
            'people_count': count,
            'features': {
                'people_count': count,
                'density_per_frame': round(density, 4),
                'avg_speed': round(avg_speed, 2),
                'growth_rate': round(growth_rate, 2),
                'is_panic': panic,
                'volatility': round(float(np.std(list(self._count_history))), 2)
                    if len(self._count_history) > 1 else 0.0
            },
            'progress': round(progress, 3),
            'scenario': self.scenario_name,
            'is_running': self.is_running
        }

    def _render_frame(self, detections: List[Dict],
                      panic: bool = False) -> np.ndarray:
        """Render a synthetic video frame with simulated people."""
        # Background
        if panic:
            frame = np.full((self.frame_height, self.frame_width, 3),
                            (30, 25, 40), dtype=np.uint8)
        else:
            frame = np.full((self.frame_height, self.frame_width, 3),
                            (45, 42, 38), dtype=np.uint8)

        # Draw subtle grid
        for x in range(0, self.frame_width, self.frame_width // 3):
            cv2.line(frame, (x, 0), (x, self.frame_height),
                     (60, 58, 55), 1)
        for y in range(0, self.frame_height, self.frame_height // 3):
            cv2.line(frame, (0, y), (self.frame_width, y),
                     (60, 58, 55), 1)

        # Draw zone labels
        zones = ['A1', 'A2', 'A3', 'B1', 'B2', 'B3', 'C1', 'C2', 'C3']
        zone_w = self.frame_width // 3
        zone_h = self.frame_height // 3
        for idx, label in enumerate(zones):
            r, c = idx // 3, idx % 3
            cx = c * zone_w + zone_w // 2 - 10
            cy = r * zone_h + 20
            cv2.putText(frame, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 78, 75), 1)

        # Draw people as colored rectangles
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            # Color based on position density
            color = (80, 200, 120) if not panic else (80, 80, 220)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Draw head circle
            cx, cy = det['center']
            cv2.circle(frame, (cx, y1 + 8), 6, color, -1)

        # HUD overlay
        count = len(detections)
        status_color = (80, 200, 120)
        if count > 30:
            status_color = (60, 60, 220)
        elif count > 15:
            status_color = (60, 180, 240)

        cv2.putText(frame, f"SIM: {self.scenario_name}",
                    (10, self.frame_height - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        cv2.putText(frame, f"Count: {count}",
                    (10, self.frame_height - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
        if panic:
            cv2.putText(frame, "!! PANIC MODE !!",
                        (self.frame_width // 2 - 80, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def generate_frames(self, scenario: str = 'normal') -> Generator:
        """
        Generator that yields frame data for a complete scenario.

        Usage:
            sim = CrowdSimulator()
            for data in sim.generate_frames('gradual_buildup'):
                frame = data['frame']
                detections = data['detections']
        """
        self.start_scenario(scenario)
        while self.is_running:
            yield self.step()

    def get_available_scenarios(self) -> Dict[str, str]:
        """List available simulation scenarios."""
        return {k: v['description'] for k, v in SCENARIOS.items()}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sim = CrowdSimulator(fps=10)

    print("Available scenarios:")
    for name, desc in sim.get_available_scenarios().items():
        print(f"  {name}: {desc}")

    print("\nRunning 'sudden_surge' scenario …")
    frame_count = 0
    for data in sim.generate_frames('sudden_surge'):
        frame_count += 1
        if frame_count % 50 == 0:
            f = data['features']
            print(f"  Frame {data['frame_number']}: "
                  f"count={f['people_count']}, "
                  f"speed={f['avg_speed']:.1f}, "
                  f"growth={f['growth_rate']:.2f}")

    print(f"Scenario complete — {frame_count} frames generated")
