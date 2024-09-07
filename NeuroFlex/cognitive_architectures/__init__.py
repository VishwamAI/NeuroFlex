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
    'create_cdstdp'
]

# Constants for self-healing and adaptive algorithms
PERFORMANCE_THRESHOLD = 0.8
UPDATE_INTERVAL = 86400  # 24 hours in seconds
LEARNING_RATE_ADJUSTMENT = 0.1
MAX_HEALING_ATTEMPTS = 5
CONSCIOUSNESS_BROADCAST_INTERVAL = 100  # milliseconds
