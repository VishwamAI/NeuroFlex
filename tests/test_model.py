# Unit tests for the tokenization and self-correcting words features in the NeuroFlex model

import unittest
from NeuroFlex.model import NeuroFlex

class TestNeuroFlexTextProcessing(unittest.TestCase):
    def setUp(self):
        # Initialize the NeuroFlex model
        self.model = NeuroFlex(features=[64, 32, 10])

    def test_process_text_basic(self):
        # Test basic text processing
        text = "This is a test sentence."
        tokens = self.model.process_text(text)
        expected_tokens = ['This', 'is', 'a', 'test', 'sentence', '.']
        self.assertEqual(tokens, expected_tokens)

    def test_process_text_with_grammar_errors(self):
        # Test text processing with grammar errors
        text = "This are a test sentence."
        tokens = self.model.process_text(text)
        expected_tokens = ['This', 'is', 'a', 'test', 'sentence', '.']  # Assuming grammar correction works
        self.assertEqual(tokens, expected_tokens)

    def test_process_text_empty(self):
        # Test processing of an empty string
        text = ""
        tokens = self.model.process_text(text)
        expected_tokens = []
        self.assertEqual(tokens, expected_tokens)

    def test_process_text_special_characters(self):
        # Test processing of text with special characters
        text = "!@#$%^&*()"
        tokens = self.model.process_text(text)
        expected_tokens = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')']
        self.assertEqual(tokens, expected_tokens)

if __name__ == '__main__':
    unittest.main()
