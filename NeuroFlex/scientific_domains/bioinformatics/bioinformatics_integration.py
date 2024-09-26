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

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from typing import List, Dict

class BioinformaticsIntegration:
    def __init__(self):
        pass

    def read_sequence_file(self, file_path: str, file_format: str = "fasta") -> List[SeqRecord]:
        """
        Read sequence data from files.

        Args:
            file_path (str): Path to the sequence file.
            file_format (str): Format of the sequence file (default: "fasta").

        Returns:
            List[SeqRecord]: List of SeqRecord objects.
        """
        return list(SeqIO.parse(file_path, file_format))

    def sequence_summary(self, sequences: List[SeqRecord]) -> List[Dict]:
        """
        Generate summaries from sequences.

        Args:
            sequences (List[SeqRecord]): List of SeqRecord objects.

        Returns:
            List[Dict]: List of dictionaries containing sequence summaries.
        """
        summaries = []
        for seq in sequences:
            summary = {
                "id": seq.id,
                "length": len(seq),
                "description": seq.description,
                "gc_content": self._calculate_gc_content(seq.seq)
            }
            summaries.append(summary)
        return summaries

    def process_sequences(self, sequences: List[SeqRecord]) -> List[SeqRecord]:
        """
        Process sequences for analysis.

        Args:
            sequences (List[SeqRecord]): List of SeqRecord objects.

        Returns:
            List[SeqRecord]: List of processed SeqRecord objects.
        """
        processed_sequences = []
        for seq in sequences:
            # Example processing: Translate DNA to protein
            if self._is_dna(seq.seq):
                translated_seq = seq.seq.translate()
                processed_seq = SeqRecord(translated_seq, id=seq.id, description=f"Translated {seq.description}")
                processed_sequences.append(processed_seq)
            else:
                processed_sequences.append(seq)
        return processed_sequences

    def _calculate_gc_content(self, sequence: Seq) -> float:
        """Calculate GC content of a sequence."""
        gc_count = sequence.count('G') + sequence.count('C')
        return gc_count / len(sequence) * 100 if len(sequence) > 0 else 0

    def _is_dna(self, sequence: Seq) -> bool:
        """Check if a sequence is DNA."""
        return set(sequence.upper()).issubset({'A', 'C', 'G', 'T', 'N'})
