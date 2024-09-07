# import unittest
# from unittest.mock import patch, MagicMock
# from NeuroFlex.correctgrammer import correct_grammar

# class TestCorrectGrammar(unittest.TestCase):
#     @patch('NeuroFlex.correctgrammer.Gramformer')
#     def setUp(self, mock_gramformer):
#         self.mock_gf = mock_gramformer.return_value
#         self.gf = correct_grammar

#     def test_basic_correction(self):
#         input_text = "I is going to the store"
#         expected_output = "I am going to the store"
#         self.mock_gf.correct.return_value = [expected_output]
#         self.assertEqual(self.gf(input_text), expected_output)

#     def test_punctuation_correction(self):
#         input_text = "Where are you going."
#         expected_output = "Where are you going?"
#         self.mock_gf.correct.return_value = [expected_output]
#         self.assertEqual(self.gf(input_text), expected_output)

#     def test_capitalization_correction(self):
#         input_text = "the quick brown fox jumps over the lazy dog"
#         expected_output = "The quick brown fox jumps over the lazy dog"
#         self.mock_gf.correct.return_value = [expected_output]
#         self.assertEqual(self.gf(input_text), expected_output)

#     def test_no_correction_needed(self):
#         input_text = "This sentence is grammatically correct."
#         self.mock_gf.correct.return_value = [input_text]
#         self.assertEqual(self.gf(input_text), input_text)

#     def test_empty_string(self):
#         input_text = ""
#         self.mock_gf.correct.return_value = []
#         self.assertEqual(self.gf(input_text), input_text)

#     def test_special_characters(self):
#         input_text = "What is your name? My name is John!"
#         self.mock_gf.correct.return_value = [input_text]
#         self.assertEqual(self.gf(input_text), input_text)

# if __name__ == '__main__':
#     unittest.main()
