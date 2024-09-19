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
from .core_neural_networks import *
from .advanced_models import *
from .generative_models import *
from .Transformers import *
from .quantum_neural_networks import *

# Importing specialized modules
from .bci_integration import *
from .cognitive_architectures import *
from .scientific_domains import *
from .edge_ai import *
from .Prompt_Agent import *

# Importing utility and ethics modules
from .utils import *
from .ai_ethics import *

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
