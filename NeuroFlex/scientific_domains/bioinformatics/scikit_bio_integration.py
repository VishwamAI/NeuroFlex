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
        except skbio.exception.BiologicalSequenceError:
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
