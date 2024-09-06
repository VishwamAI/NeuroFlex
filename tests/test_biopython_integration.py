import unittest
from unittest.mock import patch, mock_open
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from NeuroFlex.scientific_domains import BioinformaticsTools

class TestBioinformaticsTools(unittest.TestCase):
    def setUp(self):
        self.bioinformatics = BioinformaticsTools()

    @patch('Bio.SeqIO.parse')
    def test_read_sequence_file(self, mock_seqio_parse):
        mock_seq_record = SeqRecord(Seq("ATCG"), id="test_seq")
        mock_seqio_parse.return_value = iter([mock_seq_record])

        # Test successful file reading
        with patch('builtins.open', mock_open(read_data="dummy_file_content")) as mock_file:
            result = self.bioinformatics.read_sequence_file("dummy.fasta")
            self.assertEqual(len(result), 1)
            self.assertEqual(str(result[0].seq), "ATCG")
            self.assertEqual(result[0].id, "test_seq")
            mock_file.assert_called_once_with("dummy.fasta", "r")
            mock_seqio_parse.assert_called_once_with(mock_file.return_value, "fasta")

        # Test error handling when file cannot be opened
        with patch('builtins.open', side_effect=IOError("File not found")):
            with self.assertRaises(IOError):
                self.bioinformatics.read_sequence_file("non_existent.fasta")

    def test_sequence_summary(self):
        seq_record = SeqRecord(Seq("ATCG"), id="test_seq", name="Test", description="Test sequence")
        result = self.bioinformatics.sequence_summary([seq_record])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["id"], "test_seq")
        self.assertEqual(result[0]["length"], 4)
        self.assertEqual(result[0]["description"], "Test sequence")
        self.assertAlmostEqual(result[0]["gc_content"], 50.0)

    def test_process_sequences(self):
        dna_seq = SeqRecord(Seq("ATGGCCATTA"), id="dna_seq", description="DNA sequence")
        protein_seq = SeqRecord(Seq("MAIPYKKL"), id="protein_seq", description="Protein sequence")
        sequences = [dna_seq, protein_seq]

        result = self.bioinformatics.process_sequences(sequences)
        self.assertEqual(len(result), 2)
        self.assertEqual(str(result[0].seq), "MAI")
        self.assertEqual(result[0].id, "dna_seq")
        self.assertEqual(result[0].description, "Translated DNA sequence")
        self.assertEqual(str(result[1].seq), "MAIPYKKL")
        self.assertEqual(result[1].id, "protein_seq")
        self.assertEqual(result[1].description, "Protein sequence")

    def test_calculate_gc_content(self):
        sequence = SeqRecord(Seq("ATCG"), id="test_seq")
        result = self.bioinformatics.sequence_summary([sequence])[0]['gc_content']
        self.assertAlmostEqual(result, 50.0)

if __name__ == '__main__':
    unittest.main()
