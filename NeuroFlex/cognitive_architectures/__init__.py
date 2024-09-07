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
    'CONSCIOUSNESS_BROADCAST_INTERVAL'
]
