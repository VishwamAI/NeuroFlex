import unittest
from unittest.mock import patch, MagicMock
from NeuroFlex.scientific_domains.bioinformatics.scikit_bio_integration import (
    ScikitBioIntegration,
)


class TestScikitBioIntegration(unittest.TestCase):
    def setUp(self):
        self.scikit_bio = ScikitBioIntegration()

    def test_analyze_sequence_valid(self):
        with patch("skbio.sequence.DNA") as mock_dna:
            mock_dna.return_value.gc_content.return_value = 0.5
            result = self.scikit_bio.analyze_sequence("ATCG")
            self.assertIsInstance(result, dict)
            self.assertIn("gc_content", result)
            self.assertEqual(result["gc_content"], 0.5)

    def test_analyze_sequence_invalid(self):
        with patch(
            "skbio.sequence.DNA", side_effect=ValueError("Invalid DNA sequence")
        ):
            with self.assertRaises(ValueError):
                self.scikit_bio.analyze_sequence("ATCGX")  # Invalid DNA sequence

    def test_calculate_diversity_valid(self):
        with patch("skbio.diversity.alpha.shannon") as mock_shannon:
            mock_shannon.return_value = 1.5
            result = self.scikit_bio.calculate_diversity([1, 2, 3, 4, 5])
            self.assertIsInstance(result, dict)
            self.assertIn("shannon_diversity", result)
            self.assertEqual(result["shannon_diversity"], 1.5)

    def test_calculate_diversity_invalid(self):
        with patch(
            "skbio.diversity.alpha.shannon",
            side_effect=ValueError("Input data cannot be empty"),
        ):
            with self.assertRaises(ValueError):
                self.scikit_bio.calculate_diversity([])  # Empty list

    def test_align_sequences_valid(self):
        with patch("skbio.alignment.global_pairwise_align_nucleotide") as mock_align:
            mock_alignment = MagicMock()
            mock_alignment.__getitem__.side_effect = lambda i: MagicMock(
                __str__=lambda self: "ATCG"
            )
            mock_align.return_value = (mock_alignment, 4, None)
            result = self.scikit_bio.align_sequences(["ATCG", "ATCG"])
            self.assertIsInstance(result, dict)
            self.assertIn("alignment", result)
            self.assertEqual(result["alignment"], ("ATCG", "ATCG"))
            self.assertIn("score", result)
            self.assertEqual(result["score"], 4)

    def test_align_sequences_invalid(self):
        with self.assertRaises(ValueError):
            self.scikit_bio.align_sequences(["ATCG"])  # Single sequence


if __name__ == "__main__":
    unittest.main()
