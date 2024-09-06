import unittest
from unittest.mock import patch, mock_open
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Align import MultipleSeqAlignment
from NeuroFlex.scientific_domains import BiopythonIntegration

class TestBiopythonIntegration(unittest.TestCase):
    def setUp(self):
        self.biopython = BiopythonIntegration()

    @patch('Bio.SeqIO.parse')
    def test_read_sequence(self, mock_seqio_parse):
        mock_seq_record = SeqRecord(Seq("ATCG"), id="test_seq")
        mock_seqio_parse.return_value = iter([mock_seq_record])

        # Test successful file reading
        with patch('builtins.open', mock_open(read_data="dummy_file_content")) as mock_file:
            result = self.biopython.read_sequence("dummy.fasta")
            self.assertEqual(len(result), 1)
            self.assertEqual(str(result[0].seq), "ATCG")
            self.assertEqual(result[0].id, "test_seq")
            mock_file.assert_called_once_with("dummy.fasta", "r")
            mock_seqio_parse.assert_called_once_with(mock_file.return_value, "fasta")

        # Test error handling when file cannot be opened
        with patch('builtins.open', side_effect=IOError("File not found")):
            result = self.biopython.read_sequence("non_existent.fasta")
            self.assertEqual(result, [])

    @patch('Bio.SeqIO.write')
    def test_write_sequence(self, mock_seqio_write):
        sequences = [SeqRecord(Seq("ATCG"), id="test_seq")]
        result = self.biopython.write_sequence(sequences, "output.fasta")
        self.assertTrue(result)
        mock_seqio_write.assert_called_once()

    def test_translate_sequence(self):
        dna_seq = Seq("ATGGCCATTA")
        result = self.biopython.translate_sequence(dna_seq)
        self.assertEqual(str(result), "MAI")

    @patch('Bio.Align.MultipleSeqAlignment')
    def test_align_sequences(self, mock_msa):
        mock_msa.return_value = MultipleSeqAlignment([
            SeqRecord(Seq("ATCG"), id="seq1"),
            SeqRecord(Seq("ATCG"), id="seq2")
        ])
        sequences = [SeqRecord(Seq("ATCG"), id="seq1"), SeqRecord(Seq("ATCG"), id="seq2")]
        result = self.biopython.align_sequences(sequences)
        self.assertIsInstance(result, MultipleSeqAlignment)
        self.assertEqual(len(result), 2)

    def test_get_sequence_info(self):
        seq_record = SeqRecord(Seq("ATCG"), id="test_seq", name="Test", description="Test sequence")
        result = self.biopython.get_sequence_info(seq_record)
        self.assertEqual(result["id"], "test_seq")
        self.assertEqual(result["name"], "Test")
        self.assertEqual(result["description"], "Test sequence")
        self.assertEqual(result["length"], 4)
        self.assertAlmostEqual(result["gc_content"], 50.0)

    def test_calculate_gc_content(self):
        sequence = Seq("ATCG")
        result = self.biopython.calculate_gc_content(sequence)
        self.assertAlmostEqual(result, 50.0)

if __name__ == '__main__':
    unittest.main()
