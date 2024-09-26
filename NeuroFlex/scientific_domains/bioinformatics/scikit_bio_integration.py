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

# scikit_bio_integration.py
import skbio
from skbio import DNA
from skbio.alignment import global_pairwise_align_nucleotide
from skbio.diversity import alpha

class ScikitBioIntegration:
    def __init__(self):
        pass

    def analyze_sequence(self, sequence):
        try:
            dna = DNA(sequence)
            return {
                'gc_content': dna.gc_content(),
                'length': len(dna)
            }
        except ValueError:
            raise ValueError("Invalid DNA sequence")

    def calculate_diversity(self, data):
        if not data:
            raise ValueError("Input data cannot be empty")
        return {
            'shannon_diversity': alpha.shannon(data)
        }

    def align_sequences(self, sequences):
        if len(sequences) != 2:
            raise ValueError("Exactly two sequences are required for alignment")
        alignment, score, _ = global_pairwise_align_nucleotide(DNA(sequences[0]), DNA(sequences[1]))
        return {
            'alignment': (str(alignment[0]), str(alignment[1])),
            'score': score
        }
