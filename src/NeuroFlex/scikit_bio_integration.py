# Import required modules
import skbio
from skbio.alignment import local_pairwise_align_ssw
from skbio import DNA, TabularMSA
from typing import List, Tuple

class ScikitBioIntegration:
    def __init__(self):
        pass

    def align_dna_sequences(self, sequences: List[DNA]) -> List[Tuple[float, DNA, DNA]]:
        """Align DNA sequences using local pairwise alignment (SSW) and return scores and alignments."""
        alignments = []
        for i in range(len(sequences)):
            for j in range(i + 1, len(sequences)):
                score, alignment, _ = local_pairwise_align_ssw(sequences[i], sequences[j])
                alignments.append((score, alignment, sequences[i], sequences[j]))
        return alignments

    def msa_maker(self, sequences: List[DNA]) -> TabularMSA:
        """Create a Multiple Sequence Alignment (MSA) from a list of DNA sequences."""
        msa = TabularMSA(lookup=sequences)
        return msa

    def dna_gc_content(self, sequence: DNA) -> float:
        """Calculate GC content for a DNA sequence."""
        return sequence.gc_content()
