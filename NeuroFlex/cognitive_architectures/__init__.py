"""
NeuroFlex Cognitive Architectures Module

This module implements advanced cognitive architectures, including self-healing mechanisms
and adaptive algorithms based on Global Workspace Theory (GWT) concepts.

Recent updates:
- Integrated self-healing mechanisms and adaptive learning
- Implemented GWT-inspired consciousness simulation
- Enhanced extended cognitive architectures with BCI processing
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
    'create_cognitive_model'
]

def get_cognitive_architectures_version():
    return "1.0.0"

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
    if model_type == "CognitiveArchitecture":
        return CognitiveArchitecture(*args, **kwargs)
    elif model_type == "ConsciousnessSimulation":
        return ConsciousnessSimulation(*args, **kwargs)
    elif model_type == "ExtendedCognitiveArchitecture":
        return ExtendedCognitiveArchitecture(*args, **kwargs)
    elif model_type == "CDSTDP":
        return CDSTDP(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported cognitive model type: {model_type}")

# Add any other Cognitive Architectures-specific utility functions or constants as needed
