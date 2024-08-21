import unittest
from unittest.mock import Mock, patch
import os
from NeuroFlex.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock SentencePieceProcessor
        cls.mock_sp = Mock()
        cls.mock_sp.GetPieceSize.return_value = 1000
        cls.mock_sp.bos_id.return_value = 1
        cls.mock_sp.eos_id.return_value = 2
        cls.mock_sp.pad_id.return_value = 0
        cls.mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        cls.mock_sp.DecodeIds.return_value = "Hello, world!"

        # Mock os.path.isfile to return True for the dummy path
        cls.mock_isfile = patch('os.path.isfile', return_value=True)
        cls.mock_isfile.start()

        # Patch SentencePieceProcessor in Tokenizer
        with patch('sentencepiece.SentencePieceProcessor', return_value=cls.mock_sp):
            cls.tokenizer = Tokenizer("dummy_path")

    @classmethod
    def tearDownClass(cls):
        # Stop the os.path.isfile patch
        cls.mock_isfile.stop()

    def test_encode_basic(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        self.assertEqual(tokens[0], self.tokenizer.bos_id)

    def test_encode_no_bos(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text, bos=False)
        self.assertNotEqual(tokens[0], self.tokenizer.bos_id)

    def test_encode_with_eos(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text, eos=True)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)

    def test_decode_basic(self):
        text = "Hello, world!"
        tokens = self.tokenizer.encode(text)
        decoded_text = self.tokenizer.decode(tokens)
        self.assertIsInstance(decoded_text, str)
        self.assertGreater(len(decoded_text), 0)

    def test_encode_decode_roundtrip(self):
        original_text = "This is a test sentence."
        tokens = self.tokenizer.encode(original_text)
        self.mock_sp.DecodeIds.return_value = original_text
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(original_text, decoded_text)

    def test_empty_string(self):
        text = ""
        self.mock_sp.EncodeAsIds.return_value = []
        tokens = self.tokenizer.encode(text)
        self.assertEqual(len(tokens), 1)  # Should only contain BOS token
        self.mock_sp.DecodeIds.return_value = ""
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, "")

    def test_special_tokens(self):
        self.assertIsNotNone(self.tokenizer.bos_id)
        self.assertIsNotNone(self.tokenizer.eos_id)
        self.assertIsNotNone(self.tokenizer.pad_id)

if __name__ == '__main__':
    unittest.main()
