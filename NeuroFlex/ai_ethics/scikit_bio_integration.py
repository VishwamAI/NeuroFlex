import skbio
from skbio import DNA
from skbio.alignment import global_pairwise_align_nucleotide
import numpy as np
import logging
from typing import List, Tuple, Optional
from skbio.sequence.distance import hamming

class ScikitBioIntegration:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def align_dna_sequences(self, seq1: str, seq2: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
        """
        Align two DNA sequences using global pairwise alignment.

        Args:
            seq1 (str): First DNA sequence.
            seq2 (str): Second DNA sequence.

        Returns:
            tuple: A tuple containing the aligned sequences and the alignment score.
        """
        try:
            alignment = global_pairwise_align_nucleotide(DNA(seq1), DNA(seq2))
            aligned_seq1, aligned_seq2 = alignment[0]
            score = alignment[1]
            return str(aligned_seq1), str(aligned_seq2), float(score)
        except ValueError as e:
            self.logger.error(f"Invalid DNA sequence: {str(e)}")
            return None, None, None
        except Exception as e:
            self.logger.error(f"Error in aligning DNA sequences: {str(e)}")
            return None, None, None

    def calculate_sequence_similarity(self, seq1: str, seq2: str) -> float:
        """
        Calculate the similarity between two DNA sequences.

        Args:
            seq1 (str): First DNA sequence.
            seq2 (str): Second DNA sequence.

        Returns:
            float: Similarity score between 0 and 1.
        """
        aligned_seq1, aligned_seq2, _ = self.align_dna_sequences(seq1, seq2)
        if aligned_seq1 is None or aligned_seq2 is None:
            return 0.0

        matches = sum(a == b for a, b in zip(aligned_seq1, aligned_seq2))
        similarity = matches / max(len(aligned_seq1), len(aligned_seq2))
        return similarity

    def detect_anomalies(self, sequences: List[str], threshold: float = 0.8) -> List[int]:
        """
        Detect anomalies in a set of DNA sequences.

        Args:
            sequences (list): List of DNA sequences.
            threshold (float): Similarity threshold for anomaly detection.

        Returns:
            list: Indices of sequences considered anomalies.
        """
        anomalies = []
        for i, seq in enumerate(sequences):
            similarities = [self.calculate_sequence_similarity(seq, other_seq)
                            for j, other_seq in enumerate(sequences) if i != j]
            if np.mean(similarities) < threshold:
                anomalies.append(i)
        return anomalies

    def visualize_alignment(self, seq1: str, seq2: str) -> str:
        """
        Visualize the alignment of two DNA sequences.

        Args:
            seq1 (str): First DNA sequence.
            seq2 (str): Second DNA sequence.

        Returns:
            str: A string representation of the alignment visualization.
        """
        aligned_seq1, aligned_seq2, score = self.align_dna_sequences(seq1, seq2)
        if aligned_seq1 is None or aligned_seq2 is None:
            return "Alignment failed"

        match_line = ''.join('|' if a == b else ' ' for a, b in zip(aligned_seq1, aligned_seq2))
        visualization = f"Sequence 1: {aligned_seq1}\n"
        visualization += f"            {match_line}\n"
        visualization += f"Sequence 2: {aligned_seq2}\n"
        visualization += f"Alignment score: {score}"
        return visualization

    def msa_maker(self, sequences: List[str]) -> skbio.alignment.TabularMSA:
        """
        Perform multiple sequence alignment on a list of DNA sequences.

        Args:
            sequences (List[str]): List of DNA sequences to align.

        Returns:
            skbio.alignment.TabularMSA: Multiple sequence alignment object.
        """
        dna_sequences = [skbio.DNA(seq) for seq in sequences]
        msa = skbio.alignment.TabularMSA(dna_sequences)
        return msa

    def dna_gc_content(self, sequence: str) -> float:
        """
        Calculate the GC content of a DNA sequence.

        Args:
            sequence (str): DNA sequence.

        Returns:
            float: GC content as a percentage.
        """
        sequence = sequence.upper()
        gc_count = sequence.count('G') + sequence.count('C')
        total_bases = len(sequence)
        if total_bases == 0:
            return 0.0
        return (gc_count / total_bases) * 100
