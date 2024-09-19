"""
NeuroFlex Scientific Domains Module

This module provides tools and integrations for various scientific domains, including mathematics, bioinformatics, and art.

Recent updates:
- Enhanced bioinformatics integration with new tools
- Improved support for synthetic biology insights
- Added integration with Google and IBM AI services
"""

from .math_solvers import MathSolver
from .bioinformatics.bioinformatics_integration import BioinformaticsIntegration as BioinformaticsTools
from .art_integration import ARTIntegration
from .biology.synthetic_biology_insights import SyntheticBiologyInsights
from .google_integration import GoogleIntegration
from .ibm_integration import IBMIntegration
# from .alphafold_integration import AlphaFoldIntegration  # Temporarily commented out
from .xarray_integration import XarrayIntegration

__all__ = [
    'MathSolver',
    'BioinformaticsTools',
    'ARTIntegration',
    'SyntheticBiologyInsights',
    'GoogleIntegration',
    'IBMIntegration',
    # 'AlphaFoldIntegration',  # Temporarily removed
    'XarrayIntegration',
    'get_scientific_domains_version',
    'SUPPORTED_SCIENTIFIC_DOMAINS',
    'initialize_scientific_domains',
    'create_scientific_domain_model'
]

def get_scientific_domains_version():
    return "1.0.0"

SUPPORTED_SCIENTIFIC_DOMAINS = [
    "Mathematics",
    "Bioinformatics",
    "Art",
    "Synthetic Biology",
    "Google AI",
    "IBM AI",
    "Xarray"
]

def initialize_scientific_domains():
    print("Initializing Scientific Domains Module...")
    # Add any necessary initialization code here

def create_scientific_domain_model(domain, *args, **kwargs):
    if domain == "Mathematics":
        return MathSolver(*args, **kwargs)
    elif domain == "Bioinformatics":
        return BioinformaticsTools(*args, **kwargs)
    elif domain == "Art":
        return ARTIntegration(*args, **kwargs)
    elif domain == "Synthetic Biology":
        return SyntheticBiologyInsights(*args, **kwargs)
    elif domain == "Google AI":
        return GoogleIntegration(*args, **kwargs)
    elif domain == "IBM AI":
        return IBMIntegration(*args, **kwargs)
    elif domain == "Xarray":
        return XarrayIntegration(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported scientific domain: {domain}")

# Add any other Scientific Domains-specific utility functions or constants as needed
