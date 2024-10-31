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
NeuroFlex Main Module

This module serves as the entry point for the NeuroFlex library, importing core components and specialized modules.

Recent updates:
- Incremented version to 0.1.3
- Enhanced import structure for core and specialized modules
- Added package-level functions and initialization code
"""

__version__ = "0.1.3"  # Incremented version

# Direct import of core components
from .core_neural_networks import NeuroFlex, SelfCuringAlgorithm

# Lazy loading functions for core components
def load_core_neural_networks():
    from .core_neural_networks import NeuroFlex, SelfCuringAlgorithm
    return NeuroFlex, SelfCuringAlgorithm

def load_cognitive_architectures():
    from .cognitive_architectures.long_term_memory import LongTermMemory
    return {'LongTermMemory': LongTermMemory}

# Essential imports for testing
try:
    from .cognitive_architectures.long_term_memory import LongTermMemory
except ImportError:
    LongTermMemory = None

# Deferred imports - only load when explicitly requested
_core_components = None
_cognitive_components = None

def get_core_components():
    global _core_components
    if _core_components is None:
        _core_components = load_core_neural_networks()
    return _core_components

def get_cognitive_components():
    global _cognitive_components
    if _cognitive_components is None:
        _cognitive_components = load_cognitive_architectures()
    return _cognitive_components

# Importing utility and ethics modules
from .utils import *
from .ai_ethics import *

# Define what should be imported with "from NeuroFlex import *"
__all__ = [
    'NeuroFlex',
    'SelfCuringAlgorithm',
    'core_neural_networks',
    'advanced_models',
    'generative_models',
    'Transformers',
    'quantum_neural_networks',
    'bci_integration',
    'cognitive_architectures',
    'scientific_domains',
    'edge_ai',
    'Prompt_Agent',
    'utils',
    'ai_ethics'
]

# You can add any initialization code or package-level functions here
def get_version():
    return __version__

# Add any other package-level functions or variables as needed
