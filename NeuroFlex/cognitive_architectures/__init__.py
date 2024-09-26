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
NeuroFlex Cognitive Architectures Module

This module implements advanced cognitive architectures, including self-healing mechanisms
and adaptive algorithms based on Global Workspace Theory (GWT) concepts.

Recent updates:
- Integrated self-healing mechanisms and adaptive learning
- Implemented GWT-inspired consciousness simulation
- Enhanced extended cognitive architectures with BCI processing
- Updated version to match main NeuroFlex version
"""

from .cognitive_architecture import CognitiveArchitecture, create_consciousness, create_feedback_mechanism
from .consciousness_simulation import ConsciousnessSimulation, create_consciousness_simulation
from .extended_cognitive_architectures import ExtendedCognitiveArchitecture, BCIProcessor, create_extended_cognitive_model
from .advanced_thinking import CDSTDP, create_cdstdp
from ..constants import (
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL,
    LEARNING_RATE_ADJUSTMENT,
    MAX_HEALING_ATTEMPTS,
    CONSCIOUSNESS_BROADCAST_INTERVAL
)

__all__ = [
    'CognitiveArchitecture',
    'create_consciousness',
    'create_feedback_mechanism',
    'ConsciousnessSimulation',
    'create_consciousness_simulation',
    'ExtendedCognitiveArchitecture',
    'BCIProcessor',
    'create_extended_cognitive_model',
    'CDSTDP',
    'create_cdstdp',
    'PERFORMANCE_THRESHOLD',
    'UPDATE_INTERVAL',
    'LEARNING_RATE_ADJUSTMENT',
    'MAX_HEALING_ATTEMPTS',
    'CONSCIOUSNESS_BROADCAST_INTERVAL',
    'get_cognitive_architectures_version',
    'SUPPORTED_COGNITIVE_MODELS',
    'initialize_cognitive_architectures',
    'create_cognitive_model',
    'validate_cognitive_model_config'
]

def get_cognitive_architectures_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_COGNITIVE_MODELS = [
    "CognitiveArchitecture",
    "ConsciousnessSimulation",
    "ExtendedCognitiveArchitecture",
    "CDSTDP"
]

def initialize_cognitive_architectures():
    print("Initializing Cognitive Architectures Module...")
    # Add any necessary initialization code here

def create_cognitive_model(model_type, *args, **kwargs):
    if model_type not in SUPPORTED_COGNITIVE_MODELS:
        raise ValueError(f"Unsupported cognitive model type: {model_type}")

    if model_type == "CognitiveArchitecture":
        return CognitiveArchitecture(*args, **kwargs)
    elif model_type == "ConsciousnessSimulation":
        return ConsciousnessSimulation(*args, **kwargs)
    elif model_type == "ExtendedCognitiveArchitecture":
        return ExtendedCognitiveArchitecture(*args, **kwargs)
    elif model_type == "CDSTDP":
        return CDSTDP(*args, **kwargs)

def validate_cognitive_model_config(config):
    """
    Validate the configuration for a cognitive model.
    """
    required_keys = ['model_type', 'input_size', 'hidden_size', 'output_size']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['model_type'] not in SUPPORTED_COGNITIVE_MODELS:
        raise ValueError(f"Unsupported cognitive model type: {config['model_type']}")

    return True

# Add any other Cognitive Architectures-specific utility functions or constants as needed
