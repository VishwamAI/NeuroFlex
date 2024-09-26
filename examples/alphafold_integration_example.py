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
NeuroFlex AlphaFold Integration Example

This script demonstrates the usage of AlphaFold integration in NeuroFlex.
It includes initialization, configuration, and basic operations.
"""

import os
from NeuroFlex.scientific_domains.bioinformatics import AlphaFoldIntegration
from NeuroFlex.utils import data_loader

def demonstrate_alphafold_integration():
    print("Demonstrating AlphaFold Integration:")

    # Ensure ALPHAFOLD_PATH is set
    alphafold_path = os.environ.get('ALPHAFOLD_PATH')
    if not alphafold_path:
        raise EnvironmentError("ALPHAFOLD_PATH environment variable is not set.")

    # Initialize AlphaFold integration
    alphafold = AlphaFoldIntegration(alphafold_path=alphafold_path)

    # Load example protein sequence
    protein_sequence = data_loader.load_example_protein_sequence()
    print(f"Example protein sequence: {protein_sequence[:50]}...")

    # Predict protein structure
    predicted_structure = alphafold.predict_structure(protein_sequence)
    print(f"Predicted structure shape: {predicted_structure.shape}")

    # Analyze predicted structure
    confidence_score = alphafold.analyze_structure(predicted_structure)
    print(f"Structure prediction confidence score: {confidence_score}")

    # Visualize structure (in a real scenario, this would save or display an image)
    alphafold.visualize_structure(predicted_structure, output_path="predicted_structure.png")
    print("Structure visualization saved as 'predicted_structure.png'")

def main():
    demonstrate_alphafold_integration()

if __name__ == "__main__":
    main()
