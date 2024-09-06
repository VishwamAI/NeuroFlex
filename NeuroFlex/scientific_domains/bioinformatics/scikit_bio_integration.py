# scikit_bio_integration.py

import skbio
from skbio import DNA
from skbio.diversity.alpha import shannon
from skbio.alignment import global_pairwise_align_nucleotide

class ScikitBioIntegration:
    def __init__(self):
        pass

    def analyze_sequence(self, sequence_string):
        """
        Perform basic analysis on a DNA sequence.

        :param sequence_string: A string representing a DNA sequence.
        :return: DNA object
        """
        try:
            sequence = DNA(sequence_string)
            print(f"Sequence: {sequence}")
            print(f"GC Content: {sequence.gc_content():.2f}")
            print(f"Is valid: {sequence.is_valid()}")
            return sequence
        except Exception as e:
            print(f"Error analyzing sequence: {e}")
            return None

    def calculate_diversity(self, counts):
        """
        Calculate Shannon diversity index from species counts.

        :param counts: A list or array of species counts.
        :return: Shannon diversity index.
        """
        try:
            diversity_index = shannon(counts)
            print(f"Shannon Diversity Index: {diversity_index}")
            return diversity_index
        except Exception as e:
            print(f"Error calculating diversity: {e}")
            return None

    def align_sequences(self, sequence_1, sequence_2):
        """
        Align two DNA sequences using global pairwise alignment.

        :param sequence_1: A string representing the first DNA sequence.
        :param sequence_2: A string representing the second DNA sequence.
        :return: The aligned sequences.
        """
        try:
            seq1 = DNA(sequence_1)
            seq2 = DNA(sequence_2)
            alignment, score, _ = global_pairwise_align_nucleotide(seq1, seq2)
            print(f"Alignment score: {score}")
            print(f"Aligned sequences:\n{alignment}")
            return alignment, score
        except Exception as e:
            print(f"Error aligning sequences: {e}")
            return None
