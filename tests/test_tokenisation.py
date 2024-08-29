import unittest
from NeuroFlex.tokenisation import tokenize_text

class TestTokenisation(unittest.TestCase):
    def test_basic_tokenization(self):
        text = "Hello, world! This is a test."
        expected = ['Hello', ',', 'world', '!', 'This', 'is', 'a', 'test', '.']
        self.assertEqual(tokenize_text(text), expected)

    def test_empty_string(self):
        self.assertEqual(tokenize_text(""), [])

    def test_special_characters(self):
        text = "!@#$%^&*()_+"
        expected = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '_+']
        self.assertEqual(tokenize_text(text), expected)

    def test_numbers(self):
        text = "I have 3 apples and 2.5 oranges."
        expected = ['I', 'have', '3', 'apples', 'and', '2.5', 'oranges', '.']
        self.assertEqual(tokenize_text(text), expected)

    def test_contractions(self):
        text = "I'm can't won't"
        expected = ['I', "'m", 'ca', "n't", 'wo', "n't"]
        self.assertEqual(tokenize_text(text), expected)

if __name__ == '__main__':
    unittest.main()
