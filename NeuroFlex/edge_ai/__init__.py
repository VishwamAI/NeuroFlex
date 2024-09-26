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
    'MAX_HEALING_ATTEMPTS',
    'get_edge_ai_version',
    'SUPPORTED_EDGE_AI_TECHNIQUES',
    'initialize_edge_ai',
    'create_edge_ai_model'
]

def get_edge_ai_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_EDGE_AI_TECHNIQUES = [
    "Model Compression",
    "Quantization",
    "Pruning",
    "Knowledge Distillation",
    "Federated Learning",
    "Neuromorphic Computing"  # Added as it's part of this module
]

def initialize_edge_ai():
    print("Initializing Edge AI Module...")
    # Add any necessary initialization code here

def create_edge_ai_model(technique, *args, **kwargs):
    if technique in SUPPORTED_EDGE_AI_TECHNIQUES:
        if technique == "Model Compression":
            return EdgeAIOptimization.compress_model(*args, **kwargs)
        elif technique == "Quantization":
            return EdgeAIOptimization.quantize_model(*args, **kwargs)
        elif technique == "Neuromorphic Computing":
            return NeuromorphicComputing.create_neuromorphic_model(*args, **kwargs)
        # Add more techniques as needed
    else:
        raise ValueError(f"Unsupported Edge AI technique: {technique}")

# Add any other Edge AI-specific utility functions or constants as needed
