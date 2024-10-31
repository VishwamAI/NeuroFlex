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

"""
Mock AlphaFold Integration Module

This module provides a mock implementation of the AlphaFold integration for testing purposes.
It maintains the same interface as the real AlphaFold integration but returns mock data.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

class AlphaFoldIntegration:
    """Mock implementation of AlphaFold integration."""

    def __init__(self, model=None, model_params=None, config=None):
        """Initialize mock AlphaFold integration.

        Args:
            model: Mock model object
            model_params: Mock model parameters
            config: Mock configuration
        """
        self.model = model
        self.model_params = model_params
        self.config = config
        self.feature_dict = None
        self._is_ready = False

    def is_model_ready(self):
        """Check if mock model is ready.

        Returns:
            bool: True if model is ready, False otherwise
        """
        if not all([self.model, self.model_params, self.config, self.feature_dict]):
            logger.warning("Mock AlphaFold model not ready: missing components")
            return False
        return self._is_ready

    def prepare_features(self, sequence):
        """Prepare mock features for prediction.

        Args:
            sequence (str): Input protein sequence

        Returns:
            dict: Mock feature dictionary
        """
        if not sequence:
            raise ValueError("Input sequence cannot be empty")

        self.feature_dict = {
            'aatype': np.zeros((len(sequence), 21)),
            'residue_index': np.arange(len(sequence)),
            'seq_length': np.array([len(sequence)]),
            'sequence': np.array(list(sequence))
        }
        return self.feature_dict

    def predict_structure(self, sequence):
        """Predict protein structure from sequence.

        Args:
            sequence (str): Input protein sequence

        Returns:
            dict: Mock prediction results
        """
        if not self.is_model_ready():
            raise RuntimeError("Mock AlphaFold model not ready")

        # Return mock prediction results
        seq_length = len(sequence)
        return {
            'plddt': np.random.uniform(50, 90, size=(seq_length,)),
            'predicted_aligned_error': np.random.uniform(0, 10, size=(seq_length, seq_length)),
            'ptm': np.random.uniform(0.7, 0.9),
            'max_predicted_aligned_error': 10.0
        }

    def get_plddt_scores(self, logits):
        """Get pLDDT scores from logits.

        Args:
            logits (numpy.ndarray): Input logits array

        Returns:
            numpy.ndarray: Mock pLDDT scores
        """
        if not isinstance(logits, np.ndarray):
            raise TypeError("Logits must be a numpy array")

        if logits.size == 0:
            raise ValueError("Empty logits array")

        if np.any(np.isnan(logits)):
            raise ValueError("NaN values in logits")

        # Return mock pLDDT scores
        return np.random.uniform(50, 90, size=logits.shape[:-1])

    def get_predicted_aligned_error(self, prediction_result):
        """Get predicted aligned error from results.

        Args:
            prediction_result (dict): Prediction results dictionary

        Returns:
            numpy.ndarray: Mock predicted aligned error matrix
        """
        if not isinstance(prediction_result, dict):
            raise TypeError("Prediction result must be a dictionary")

        if 'predicted_aligned_error' not in prediction_result:
            raise KeyError("Missing predicted_aligned_error in results")

        pae = prediction_result['predicted_aligned_error']

        if not isinstance(pae, np.ndarray):
            raise TypeError("Predicted aligned error must be a numpy array")


        if pae.ndim != 2:
            raise ValueError("Predicted aligned error must be a 2D array")

        if pae.shape[0] != pae.shape[1]:
            raise ValueError("Predicted aligned error must be a square matrix")

        return pae

    def setup_model(self, model=None, model_params=None, config=None):
        """Set up mock model with given parameters.

        Args:
            model: Mock model object
            model_params: Mock model parameters
            config: Mock configuration
        """
        self.model = model or self.model
        self.model_params = model_params or self.model_params
        self.config = config or self.config
        self._is_ready = all([self.model, self.model_params, self.config])
