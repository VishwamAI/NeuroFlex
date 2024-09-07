"""
NeuroFlex Edge AI Module

This module implements edge AI optimization techniques and neuromorphic computing
with self-healing capabilities and adaptive algorithms. It integrates advanced
optimization methods for edge devices with neuromorphic computing concepts to create
efficient and adaptive edge AI solutions.

Recent updates:
- Integrated self-healing mechanisms and adaptive learning
- Implemented performance monitoring and diagnosis
- Enhanced edge AI optimization with advanced techniques
"""

from .edge_ai_optimization import EdgeAIOptimization
from .neuromorphic_computing import NeuromorphicComputing
from ..constants import (
    PERFORMANCE_THRESHOLD,
    UPDATE_INTERVAL,
    LEARNING_RATE_ADJUSTMENT,
    MAX_HEALING_ATTEMPTS
)

__all__ = [
    'EdgeAIOptimization',
    'NeuromorphicComputing',
    'PERFORMANCE_THRESHOLD',
    'UPDATE_INTERVAL',
    'LEARNING_RATE_ADJUSTMENT',
    'MAX_HEALING_ATTEMPTS'
]
