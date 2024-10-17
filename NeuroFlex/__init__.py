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

# Importing core components
from .core_neural_networks.model import SelfCuringAlgorithm
from .advanced_models.multi_modal_learning import MultiModalLearning
from .generative_models.generative_ai import GenerativeAIModel
from .Transformers.unified_transformer import UnifiedTransformer
from .quantum_neural_networks.quantum_module import AdvancedQuantumModel

# Importing specialized modules
from .bci_integration.bci_processing import BCIProcessor
# ConsciousnessSimulation import removed to avoid circular dependency
from .scientific_domains.bioinformatics.bioinformatics_integration import BioinformaticsIntegration
from .edge_ai.edge_ai_optimization import EdgeAIOptimization

# Importing utility and ethics modules
from .utils.utils import tokenize_text
from .ai_ethics.advanced_security_agent import AdvancedSecurityAgent

class NeuroFlex:
    def __init__(self):
        self.self_curing_algorithm = SelfCuringAlgorithm()
        self.multi_modal_learning = MultiModalLearning()
        self.generative_ai_model = GenerativeAIModel()
        self.unified_transformer = UnifiedTransformer()
        self.advanced_quantum_model = AdvancedQuantumModel()
        self.bci_processor = BCIProcessor()
        self.bioinformatics_integration = BioinformaticsIntegration()
        self.edge_ai_optimization = EdgeAIOptimization()
        self.advanced_security_agent = AdvancedSecurityAgent()

    @property
    def consciousness_simulation(self):
        from .cognitive_architectures.consciousness_simulation import ConsciousnessSimulation
        return ConsciousnessSimulation()

# Define what should be imported with "from NeuroFlex import *"
__all__ = [
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
