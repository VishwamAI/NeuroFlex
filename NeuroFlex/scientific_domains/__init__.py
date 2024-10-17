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
NeuroFlex Scientific Domains Module

This module provides tools and integrations for various scientific domains, including mathematics, bioinformatics, and art.

Recent updates:
- Enhanced bioinformatics integration with new tools
- Improved support for synthetic biology insights
- Added integration with Google and IBM AI services
- Updated version to match main NeuroFlex version
"""

from .math_solvers import MathSolver
from .bioinformatics.bioinformatics_integration import BioinformaticsIntegration as BioinformaticsTools
from .art_integration import ARTIntegration
from .biology.synthetic_biology_insights import SyntheticBiologyInsights
from .google_integration import GoogleIntegration
from .ibm_integration import IBMIntegration
from .alphafold_integration import AlphaFoldIntegration  # Temporarily commented out
from .xarray_integration import XarrayIntegration

__all__ = [
    'MathSolver',
    'BioinformaticsTools',
    'ARTIntegration',
    'SyntheticBiologyInsights',
    'GoogleIntegration',
    'IBMIntegration',
    'AlphaFoldIntegration',  # Temporarily removed
    'XarrayIntegration',
    'get_scientific_domains_version',
    'SUPPORTED_SCIENTIFIC_DOMAINS',
    'initialize_scientific_domains',
    'create_scientific_domain_model',
    'validate_scientific_domain_config'
]

def get_scientific_domains_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

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
    if domain not in SUPPORTED_SCIENTIFIC_DOMAINS:
        raise ValueError(f"Unsupported scientific domain: {domain}")

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

def validate_scientific_domain_config(config):
    """
    Validate the configuration for a scientific domain model.
    """
    required_keys = ['domain', 'parameters']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['domain'] not in SUPPORTED_SCIENTIFIC_DOMAINS:
        raise ValueError(f"Unsupported scientific domain: {config['domain']}")

    return True

# Add any other Scientific Domains-specific utility functions or constants as needed
