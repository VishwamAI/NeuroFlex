import logging
from typing import Dict, Any, List
import numpy as np
from collections import deque

class ModelMonitor:
    """
    ModelMonitor class for monitoring and analyzing the performance and health of a model.

    This class provides functionality to track model performance over time, assess trends,
    and evaluate the stability of the model based on its performance history.
    """

    def __init__(self, performance_window: int = 100):
        """
        Initialize the ModelMonitor.

        Args:
            performance_window (int): The number of recent performance entries to keep in memory.
                                      Default is 100.

        Attributes:
            logger (logging.Logger): Logger for the ModelMonitor.
            performance_history (collections.deque): A fixed-size queue to store recent performance data.
            health_history (list): A list to store the history of health check results.
        """
        self.logger = logging.getLogger(__name__)
        self.performance_history = deque(maxlen=performance_window)
        self.health_history = []

    def setup(self):
        """Set up the ModelMonitor."""
        self.logger.info("Setting up ModelMonitor...")

    def update(self, state: Any, action: Any, reward: float, next_state: Any, done: bool):
        """
        Update the performance history with a new reward.

        Args:
            state (Any): The current state (not used in this method).
            action (Any): The action taken (not used in this method).
            reward (float): The reward received for the action.
            next_state (Any): The resulting state (not used in this method).
            done (bool): Whether the episode is done (not used in this method).
        """
        self.performance_history.append(reward)

    def check_health(self) -> Dict[str, Any]:
        """
        Perform a health check on the model.

        Returns:
            Dict[str, Any]: A dictionary containing health status information including
                            average reward, performance trend, and model stability.
        """
        avg_performance = np.mean(self.performance_history) if self.performance_history else 0
        health_status = {
            "average_reward": avg_performance,
            "performance_trend": self._calculate_trend(),
            "model_stability": self._assess_stability()
        }
        self.health_history.append(health_status)
        return health_status

    def _calculate_trend(self) -> str:
        """
        Calculate the performance trend based on recent history.

        Returns:
            str: A string indicating whether the performance is "Improving", "Degrading", or "Stable".
                 Returns "Not enough data" if there's insufficient history.
        """
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
        """
        Assess the stability of the model based on performance variability.

        Returns:
            str: A string indicating the stability level: "Very Stable", "Stable",
                 "Moderately Stable", or "Unstable". Returns "Not enough data" if
                 there's insufficient history.
        """
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
        """
        Calculate the overall performance based on the entire performance history.

        Returns:
            float: The mean of all recorded performance values, or 0 if no data is available.
        """
        return np.mean(self.performance_history) if self.performance_history else 0

    def get_health_history(self) -> List[Dict[str, Any]]:
        """
        Retrieve the complete history of health check results.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing the results of a health check.
        """
        return self.health_history
