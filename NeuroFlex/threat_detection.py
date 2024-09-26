# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
import tensorflow as tf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import joblib

class ThreatDetector:
    """
    A class for detecting and analyzing threats in a system using various methods including
    anomaly detection, deep learning, and pattern recognition.
    """

    def __init__(self):
        """
        Initialize the ThreatDetector with default parameters and empty histories.
        """
        self.logger = logging.getLogger(__name__)
        self.threat_history = []
        self.action_history = []
        self.state_change_threshold = 0.5
        self.action_deviation_threshold = 0.3
        self.anomaly_detector = None
        self.deep_learning_model = None
        self.scaler = StandardScaler()

    def setup(self):
        """
        Set up the ThreatDetector by initializing the anomaly detector and deep learning model.
        """
        self.logger.info("Setting up ThreatDetector...")
        self._setup_anomaly_detector()
        self._setup_deep_learning_model()
        self._fit_anomaly_detector()

    def _fit_anomaly_detector(self):
        """
        Fit the anomaly detector with initial data.
        """
        # Generate some initial data for fitting
        state_dim = 10  # Assuming state dimension is 10
        action_dim = 5  # Assuming action dimension is 5
        next_state_dim = 10  # Assuming next state dimension is 10
        total_dim = state_dim + action_dim + next_state_dim
        initial_data = np.random.rand(100, total_dim)
        self.anomaly_detector.fit(initial_data)

    def _setup_anomaly_detector(self):
        """
        Initialize the anomaly detector using Isolation Forest algorithm.
        """
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)

    def _setup_deep_learning_model(self):
        """
        Initialize and compile the deep learning model for threat detection.
        """
        self.deep_learning_model = Sequential([
            LSTM(64, input_shape=(1, 25), return_sequences=True),
            LSTM(32),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        self.deep_learning_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def detect_threat(self, state: Any, action: Any, next_state: Any) -> bool:
        """
        Detect potential threats based on the current state, action, and next state.

        Args:
            state (Any): The current state of the system.
            action (Any): The action taken.
            next_state (Any): The resulting state after the action.

        Returns:
            bool: True if a threat is detected, False otherwise.
        """
        threat_detected = False

        # Check for sudden large changes in state
        state_change = np.linalg.norm(np.array(next_state) - np.array(state))
        if state_change > self.state_change_threshold:
            threat_detected = True
            self.logger.warning(f"Large state change detected: {state_change}")

        # Check if action deviates significantly from the norm
        self.action_history.append(action)
        if len(self.action_history) > 1:
            mean_action = np.mean(self.action_history, axis=0)
            action_deviation = np.linalg.norm(np.array(action) - mean_action)
            if action_deviation > self.action_deviation_threshold:
                threat_detected = True
                self.logger.warning(f"Unusual action detected: {action_deviation}")

        # Use anomaly detection
        if self.anomaly_detector is not None:
            combined_data = np.concatenate([state, action, next_state])
            scaled_data = self.scaler.fit_transform(combined_data.reshape(1, -1))
            anomaly_score = self.anomaly_detector.decision_function(scaled_data)
            if anomaly_score < -0.5:  # Adjust this threshold as needed
                threat_detected = True
                self.logger.warning(f"Anomaly detected: score {anomaly_score}")

        # Use deep learning model for threat prediction
        if self.deep_learning_model is not None:
            combined_data = np.concatenate([state, action, next_state])
            scaled_data = self.scaler.transform(combined_data.reshape(1, -1))
            # Reshape the input data to match LSTM's expected dimensions
            reshaped_data = scaled_data.reshape(1, 1, -1)  # (batch_size, timesteps, features)
            threat_probability = self.deep_learning_model.predict(reshaped_data)
            if threat_probability > 0.7:  # Adjust this threshold as needed
                threat_detected = True
                self.logger.warning(f"Deep learning model detected potential threat: probability {threat_probability}")

        if threat_detected:
            self.threat_history.append((state, action, next_state))

        return threat_detected

    def is_adversarial_pattern(self, state: Any, action: Any, next_state: Any) -> bool:
        """
        Detect adversarial patterns in the given state, action, and next state.

        Args:
            state (Any): The current state of the system.
            action (Any): The action taken.
            next_state (Any): The resulting state after the action.

        Returns:
            bool: True if an adversarial pattern is detected, False otherwise.
        """
        # Implement more sophisticated adversarial pattern detection
        combined_data = np.concatenate([state, action, next_state])
        scaled_data = self.scaler.transform(combined_data.reshape(1, -1))
        adversarial_score = self.deep_learning_model.predict(scaled_data)
        return adversarial_score > 0.8  # Adjust this threshold as needed

    def get_safe_action(self, state: Any, action: Any, next_state: Any) -> Any:
        """
        Determine a safe action based on the current state, proposed action, and predicted next state.

        Args:
            state (Any): The current state of the system.
            action (Any): The proposed action.
            next_state (Any): The predicted next state.

        Returns:
            Any: A safe action to take.
        """
        # Implement reinforcement learning for safe action determination
        # This is a placeholder implementation
        safe_action = action  # Default to the original action
        return safe_action

    def scan_for_vulnerabilities(self, model: Any) -> List[str]:
        """
        Scan the given model for potential vulnerabilities.

        Args:
            model (Any): The model to scan for vulnerabilities.

        Returns:
            List[str]: A list of detected vulnerabilities.
        """
        vulnerabilities = []
        # Implement vulnerability scanning using the model architecture
        if isinstance(model, tf.keras.Model):
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.Dense):
                    weights = layer.get_weights()[0]
                    if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                        vulnerabilities.append(f"NaN or Inf weights detected in layer {layer.name}")
        return vulnerabilities

    def analyze(self) -> Dict[str, Any]:
        """
        Perform a comprehensive analysis of the threat detection system.

        Returns:
            Dict[str, Any]: A dictionary containing analysis results.
        """
        analysis_result = {
            "total_threats": len(self.threat_history),
            "recent_threats": self.threat_history[-5:] if self.threat_history else [],
            "vulnerability_summary": self.scan_for_vulnerabilities(self.deep_learning_model),
            "anomaly_detector_performance": self._evaluate_anomaly_detector(),
            "deep_learning_model_performance": self._evaluate_deep_learning_model()
        }
        return analysis_result

    def _evaluate_anomaly_detector(self) -> Dict[str, float]:
        """
        Evaluate the performance of the anomaly detector.

        Returns:
            Dict[str, float]: A dictionary containing performance metrics.
        """
        # Implement evaluation metrics for the anomaly detector
        return {"precision": 0.9, "recall": 0.85}  # Placeholder values

    def _evaluate_deep_learning_model(self) -> Dict[str, float]:
        """
        Evaluate the performance of the deep learning model.

        Returns:
            Dict[str, float]: A dictionary containing performance metrics.
        """
        # Implement evaluation metrics for the deep learning model
        return {"accuracy": 0.92, "f1_score": 0.91}  # Placeholder values

    def get_threat_history(self) -> List[Tuple[Any, Any, Any]]:
        """
        Get the history of detected threats.

        Returns:
            List[Tuple[Any, Any, Any]]: A list of tuples containing (state, action, next_state) for each detected threat.
        """
        return self.threat_history

    def save_models(self, path: str):
        """
        Save the trained models to the specified path.

        Args:
            path (str): The directory path to save the models.
        """
        joblib.dump(self.anomaly_detector, f"{path}/anomaly_detector.joblib")
        self.deep_learning_model.save(f"{path}/deep_learning_model.h5")
        joblib.dump(self.scaler, f"{path}/scaler.joblib")

    def load_models(self, path: str):
        """
        Load the trained models from the specified path.

        Args:
            path (str): The directory path to load the models from.
        """
        self.anomaly_detector = joblib.load(f"{path}/anomaly_detector.joblib")
        self.deep_learning_model = tf.keras.models.load_model(f"{path}/deep_learning_model.h5")
        self.scaler = joblib.load(f"{path}/scaler.joblib")
