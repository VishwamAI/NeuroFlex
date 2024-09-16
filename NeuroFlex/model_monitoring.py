import logging
from typing import Dict, Any, List
import numpy as np
from collections import deque

class ModelMonitor:
    def __init__(self, performance_window: int = 100):
        self.logger = logging.getLogger(__name__)
        self.performance_history = deque(maxlen=performance_window)
        self.health_history = []

    def setup(self):
        self.logger.info("Setting up ModelMonitor...")

    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        self.performance_history.append(reward)

    def check_health(self) -> Dict[str, Any]:
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0
        health_status = {
            "average_reward": avg_performance,
            "performance_trend": self._calculate_trend(),
            "model_stability": self._assess_stability()
        }
        self.health_history.append(health_status)
        return health_status

    def _calculate_trend(self) -> str:
        if len(self.performance_history) < 2:
            return "Not enough data"

        recent_avg = np.mean(list(self.performance_history)[-10:])
        overall_avg = np.mean(self.performance_history)

        if recent_avg > overall_avg * 1.1:
            return "Improving"
        elif recent_avg < overall_avg * 0.9:
            return "Degrading"
        else:
            return "Stable"

    def _assess_stability(self) -> str:
        if len(self.performance_history) < 10:
            return "Not enough data"

        std_dev = np.std(self.performance_history)
        mean = np.mean(self.performance_history)

        cv = std_dev / mean if mean != 0 else float('inf')

        if cv < 0.1:
            return "Very Stable"
        elif cv < 0.25:
            return "Stable"
        elif cv < 0.5:
            return "Moderately Stable"
        else:
            return "Unstable"

    def get_overall_performance(self) -> float:
        return np.mean(self.performance_history) if self.performance_history else 0

    def get_health_history(self) -> List[Dict[str, Any]]:
        return self.health_history
