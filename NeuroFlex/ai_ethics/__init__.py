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
NeuroFlex AI Ethics Module

This module implements ethical frameworks and guidelines for AI, including explainable AI and self-curing algorithms.

Recent updates:
- Integrated ethical principles for AI development
- Enhanced explainable AI capabilities
- Implemented self-curing reinforcement learning agents
- Updated version to match main NeuroFlex version
"""

from .ethical_framework import EthicalFramework, Guideline
from .explainable_ai import ExplainableAI
from .self_fixing_algorithms import SelfCuringRLAgent

__all__ = [
    'EthicalFramework',
    'Guideline',
    'ExplainableAI',
    'SelfCuringRLAgent',
    'get_ai_ethics_version',
    'ETHICAL_PRINCIPLES',
    'initialize_ai_ethics',
    'create_ethical_framework'
]

def get_ai_ethics_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

ETHICAL_PRINCIPLES = [
    "Beneficence",
    "Non-maleficence",
    "Autonomy",
    "Justice",
    "Explicability"
]

def initialize_ai_ethics():
    print("Initializing AI Ethics Module...")
    # Add any necessary initialization code here

def create_ethical_framework(principles=None):
    if principles is None:
        principles = ETHICAL_PRINCIPLES
    return EthicalFramework(principles)

def evaluate_ethical_compliance(model, framework=None):
    """
    Evaluate the ethical compliance of a given model using the specified ethical framework.
    If no framework is provided, a default one is created using ETHICAL_PRINCIPLES.
    """
    if framework is None:
        framework = create_ethical_framework()
    return framework.evaluate_model(model)

# Add any other AI Ethics-specific utility functions or constants as needed
