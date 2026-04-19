"""
Reinforcement Learning Crowd Control Agent
Q-learning based agent that suggests crowd management actions
based on current density, flow, and risk state per zone.
"""

import numpy as np
import json
import os
import logging
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==========================================================================
# State / Action definitions
# ==========================================================================

# Discrete density levels per zone
DENSITY_LEVELS = ['empty', 'low', 'moderate', 'high', 'critical']

# Possible control actions
ACTIONS = [
    'no_action',
    'redirect_north',
    'redirect_south',
    'redirect_east',
    'redirect_west',
    'open_alternate_path',
    'close_entry',
    'deploy_security',
    'make_announcement',
    'activate_barriers'
]

ACTION_DESCRIPTIONS = {
    'no_action': 'No action required — situation is under control.',
    'redirect_north': 'Redirect crowd flow towards the north exit/pathway.',
    'redirect_south': 'Redirect crowd flow towards the south exit/pathway.',
    'redirect_east': 'Redirect crowd flow towards the east exit/pathway.',
    'redirect_west': 'Redirect crowd flow towards the west exit/pathway.',
    'open_alternate_path': 'Open alternate pathways to distribute crowd load.',
    'close_entry': 'Temporarily close entry points to limit incoming flow.',
    'deploy_security': 'Deploy additional security personnel to manage crowd.',
    'make_announcement': 'Make public announcement to guide crowd movement.',
    'activate_barriers': 'Activate physical barriers to control crowd direction.'
}


def discretize_density(density: float) -> str:
    """Convert continuous density value to discrete level."""
    if density < 0.1:
        return 'empty'
    elif density < 0.3:
        return 'low'
    elif density < 0.6:
        return 'moderate'
    elif density < 0.85:
        return 'high'
    else:
        return 'critical'


def discretize_speed(speed: float) -> str:
    """Convert average speed to discrete level."""
    if speed < 5:
        return 'stationary'
    elif speed < 20:
        return 'slow'
    elif speed < 50:
        return 'normal'
    else:
        return 'fast'


def discretize_risk(risk_level: str) -> str:
    """Normalise risk level string."""
    return risk_level.lower() if risk_level else 'low'


# ==========================================================================
# Q-Learning Agent
# ==========================================================================

class CrowdControlAgent:
    """
    Tabular Q-learning agent for crowd management.

    State: (density_level, speed_level, risk_level, has_conflict)
    Action: one of ACTIONS
    Reward: computed from density reduction and risk mitigation.
    """

    def __init__(self, alpha: float = 0.1, gamma: float = 0.95,
                 epsilon: float = 0.15):
        """
        Args:
            alpha: Learning rate
            gamma: Discount factor
            epsilon: Exploration rate (ε-greedy)
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table: Dict[str, np.ndarray] = defaultdict(
            lambda: np.zeros(len(ACTIONS))
        )
        self.actions = ACTIONS
        self.n_actions = len(ACTIONS)
        self._episode_rewards: List[float] = []
        self.is_trained = False

        # Pre-seed with domain knowledge (heuristic rewards)
        self._seed_knowledge()

    def _state_key(self, density: str, speed: str, risk: str,
                   conflict: bool) -> str:
        """Create hashable state key."""
        return f"{density}|{speed}|{risk}|{int(conflict)}"

    def _seed_knowledge(self):
        """Pre-populate Q-table with domain heuristics for faster convergence."""
        rules = [
            # Critical density → close entry + deploy security
            ('critical', 'fast', 'high', True,
             {'close_entry': 8, 'deploy_security': 7, 'activate_barriers': 6,
              'make_announcement': 5}),
            ('critical', 'normal', 'high', False,
             {'close_entry': 7, 'redirect_north': 5, 'deploy_security': 6}),
            # High density
            ('high', 'normal', 'medium', False,
             {'open_alternate_path': 6, 'redirect_east': 4,
              'make_announcement': 3}),
            ('high', 'fast', 'high', True,
             {'deploy_security': 7, 'close_entry': 6,
              'activate_barriers': 5}),
            # Moderate density
            ('moderate', 'normal', 'low', False,
             {'no_action': 5}),
            ('moderate', 'fast', 'medium', True,
             {'make_announcement': 4, 'redirect_south': 3}),
            # Low / empty
            ('low', 'slow', 'low', False, {'no_action': 5}),
            ('empty', 'stationary', 'low', False, {'no_action': 5}),
        ]

        for density, speed, risk, conflict, action_rewards in rules:
            key = self._state_key(density, speed, risk, conflict)
            for action, reward in action_rewards.items():
                idx = self.actions.index(action)
                self.q_table[key][idx] = reward

        self.is_trained = True

    # ------------------------------------------------------------------
    # Core RL methods
    # ------------------------------------------------------------------

    def choose_action(self, state_key: str) -> int:
        """ε-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state_key]))

    def update(self, state_key: str, action_idx: int, reward: float,
               next_state_key: str):
        """Q-learning update rule."""
        best_next = np.max(self.q_table[next_state_key])
        current = self.q_table[state_key][action_idx]
        self.q_table[state_key][action_idx] = current + self.alpha * (
            reward + self.gamma * best_next - current
        )

    def compute_reward(self, prev_density: float, curr_density: float,
                       prev_risk: str, curr_risk: str, action: str) -> float:
        """
        Compute reward based on state transition.
        Positive rewards for reducing density/risk.
        """
        reward = 0.0

        # Density reduction bonus
        density_change = prev_density - curr_density
        reward += density_change * 10

        # Risk level change
        risk_map = {'low': 0, 'medium': 1, 'high': 2}
        risk_diff = risk_map.get(prev_risk, 0) - risk_map.get(curr_risk, 0)
        reward += risk_diff * 5

        # Penalty for unnecessary intervention when things are fine
        if prev_density < 0.3 and curr_density < 0.3 and action != 'no_action':
            reward -= 2

        # Bonus for correct no-action
        if prev_density < 0.2 and action == 'no_action':
            reward += 1

        return reward

    # ------------------------------------------------------------------
    # Training (simulated episodes)
    # ------------------------------------------------------------------

    def train_simulated(self, n_episodes: int = 1000):
        """
        Train agent using simulated environment transitions.
        """
        logger.info(f"Training RL agent for {n_episodes} episodes …")

        for episode in range(n_episodes):
            # Random initial state
            density = np.random.choice(DENSITY_LEVELS)
            speed = np.random.choice(['stationary', 'slow', 'normal', 'fast'])
            risk = np.random.choice(['low', 'medium', 'high'])
            conflict = np.random.random() < 0.2

            total_reward = 0.0
            density_val = {'empty': 0.05, 'low': 0.2, 'moderate': 0.45,
                           'high': 0.72, 'critical': 0.92}[density]

            for step in range(20):
                state_key = self._state_key(density, speed, risk, conflict)
                action_idx = self.choose_action(state_key)
                action = self.actions[action_idx]

                # Simulate environment transition
                new_density_val, new_risk = self._simulate_transition(
                    density_val, risk, action, conflict
                )
                new_density = discretize_density(new_density_val)
                new_speed = speed  # Simplified
                new_conflict = np.random.random() < 0.15

                reward = self.compute_reward(
                    density_val, new_density_val, risk, new_risk, action
                )

                next_state_key = self._state_key(
                    new_density, new_speed, new_risk, new_conflict
                )
                self.update(state_key, action_idx, reward, next_state_key)

                density = new_density
                density_val = new_density_val
                risk = new_risk
                conflict = new_conflict
                total_reward += reward

            self._episode_rewards.append(total_reward)

            # Decay exploration
            self.epsilon = max(0.05, self.epsilon * 0.999)

        self.is_trained = True
        avg_reward = np.mean(self._episode_rewards[-100:])
        logger.info(f"RL training complete — avg reward (last 100): {avg_reward:.2f}")

        return {
            'episodes': n_episodes,
            'avg_reward_last100': float(avg_reward),
            'q_table_size': len(self.q_table)
        }

    def _simulate_transition(self, density: float, risk: str,
                             action: str, conflict: bool
                             ) -> Tuple[float, str]:
        """Simulate how the environment responds to an action."""
        new_density = density
        new_risk = risk
        noise = np.random.normal(0, 0.03)

        if action == 'no_action':
            new_density += noise + 0.01  # Slight natural increase
        elif action in ('redirect_north', 'redirect_south',
                        'redirect_east', 'redirect_west'):
            new_density -= 0.08 + noise
            if risk == 'high':
                new_risk = 'medium'
        elif action == 'open_alternate_path':
            new_density -= 0.12 + noise
            if risk == 'high':
                new_risk = 'medium'
        elif action == 'close_entry':
            new_density -= 0.06 + noise
            if risk == 'high':
                new_risk = 'medium'
        elif action == 'deploy_security':
            new_density -= 0.04 + noise
            if conflict:
                new_density -= 0.05
            new_risk = 'medium' if risk == 'high' else risk
        elif action == 'make_announcement':
            new_density -= 0.05 + noise
        elif action == 'activate_barriers':
            new_density -= 0.1 + noise
            if risk == 'high':
                new_risk = 'medium'

        new_density = max(0, min(1, new_density))

        # Risk follows density
        if new_density > 0.85:
            new_risk = 'high'
        elif new_density > 0.5:
            new_risk = max(new_risk, 'medium') if new_risk != 'high' else 'medium'

        return new_density, new_risk

    # ------------------------------------------------------------------
    # Inference (get suggestions)
    # ------------------------------------------------------------------

    def get_suggestions(self, density: float, avg_speed: float,
                        risk_level: str, direction_conflict: bool,
                        top_k: int = 3) -> List[Dict]:
        """
        Get top-K crowd control action suggestions.

        Args:
            density: Normalised density [0, 1]
            avg_speed: Average crowd speed (px/sec)
            risk_level: 'low', 'medium', or 'high'
            direction_conflict: Whether direction conflict is detected
            top_k: Number of suggestions to return

        Returns:
            List of suggestion dicts with action, description, confidence
        """
        d_level = discretize_density(density)
        s_level = discretize_speed(avg_speed)
        r_level = discretize_risk(risk_level)
        state_key = self._state_key(d_level, s_level, r_level,
                                     direction_conflict)

        q_values = self.q_table[state_key]
        sorted_indices = np.argsort(q_values)[::-1]

        # Normalise Q-values into confidence scores
        q_max = q_values.max()
        q_min = q_values.min()
        q_range = q_max - q_min if q_max != q_min else 1.0

        suggestions = []
        for idx in sorted_indices[:top_k]:
            action = self.actions[idx]
            confidence = (q_values[idx] - q_min) / q_range if q_range > 0 else 0.5

            suggestions.append({
                'action': action,
                'description': ACTION_DESCRIPTIONS.get(action, action),
                'confidence': round(float(confidence), 3),
                'q_value': round(float(q_values[idx]), 3),
                'priority': len(suggestions) + 1
            })

        return suggestions

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, filepath: str):
        """Save Q-table to JSON file."""
        data = {
            'alpha': self.alpha,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'q_table': {k: v.tolist() for k, v in self.q_table.items()},
            'actions': self.actions
        }
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"RL agent saved to {filepath}")

    def load(self, filepath: str):
        """Load Q-table from JSON file."""
        if not os.path.exists(filepath):
            logger.warning(f"No RL agent file at {filepath}")
            return
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.alpha = data['alpha']
        self.gamma = data['gamma']
        self.epsilon = data['epsilon']
        self.q_table = defaultdict(
            lambda: np.zeros(len(ACTIONS)),
            {k: np.array(v) for k, v in data['q_table'].items()}
        )
        self.is_trained = True
        logger.info(f"RL agent loaded from {filepath}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    agent = CrowdControlAgent()

    # Train on simulated environment
    result = agent.train_simulated(n_episodes=500)
    print(f"Training: {result}")

    # Test suggestions
    print("\n--- Scenario 1: Low risk ---")
    s1 = agent.get_suggestions(density=0.2, avg_speed=10, risk_level='low',
                                direction_conflict=False)
    for s in s1:
        print(f"  [{s['priority']}] {s['action']}: {s['description']} "
              f"(conf: {s['confidence']})")

    print("\n--- Scenario 2: High risk + conflict ---")
    s2 = agent.get_suggestions(density=0.88, avg_speed=45, risk_level='high',
                                direction_conflict=True)
    for s in s2:
        print(f"  [{s['priority']}] {s['action']}: {s['description']} "
              f"(conf: {s['confidence']})")
