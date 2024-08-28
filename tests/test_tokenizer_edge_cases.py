import unittest
from unittest.mock import Mock, patch
import os
from NeuroFlex.tokenizer import Tokenizer

class TestTokenizerEdgeCases(unittest.TestCase):
    def setUp(self):
        # Mock SentencePieceProcessor
        self.mock_sp = Mock()
        self.mock_sp.GetPieceSize.return_value = 1000
        self.mock_sp.bos_id.return_value = 1
        self.mock_sp.eos_id.return_value = 2
        self.mock_sp.pad_id.return_value = 0
        self.mock_sp.unk_id.return_value = 3
        self.mock_sp.PieceToId.return_value = 4  # For <space> token

        # Reset mock calls for each test
        self.mock_sp.EncodeAsIds.reset_mock()
        self.mock_sp.DecodeIds.reset_mock()

        # Mock os.path.isfile to return True for the dummy path
        self.mock_isfile = patch('os.path.isfile', return_value=True)
        self.mock_isfile.start()

        # Patch SentencePieceProcessor in Tokenizer
        self.mock_sp_processor = patch('sentencepiece.SentencePieceProcessor', return_value=self.mock_sp)
        self.mock_sp_processor.start()

        self.tokenizer = Tokenizer("dummy_path")

    def tearDown(self):
        # Stop all patches
        self.mock_isfile.stop()
        self.mock_sp_processor.stop()

    def test_very_long_text(self):
        # Test with a very long text (e.g., 1 million characters)
        long_text = "a" * 1_000_000
        self.mock_sp.EncodeAsIds.return_value = list(range(100000))  # Simulate 100,000 tokens
        tokens = self.tokenizer.encode(long_text)
        self.assertGreater(len(tokens), 100000)  # Should be greater due to BOS and EOS tokens
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.mock_sp.EncodeAsIds.assert_called_once_with(long_text)

    def test_text_with_special_characters(self):
        special_char_text = "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        self.mock_sp.EncodeAsIds.return_value = list(range(10, 40))  # Simulate 30 tokens
        tokens = self.tokenizer.encode(special_char_text)
        self.assertEqual(len(tokens), 32)  # 30 tokens + BOS + EOS
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.mock_sp.EncodeAsIds.assert_called_once_with(special_char_text)

    def test_multilingual_text(self):
        multilingual_text = "Hello こんにちは Bonjour नमस्ते 你好"
        self.mock_sp.EncodeAsIds.return_value = list(range(10, 20))  # Simulate 10 tokens
        tokens = self.tokenizer.encode(multilingual_text)
        self.assertEqual(len(tokens), 12)  # 10 tokens + BOS + EOS
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.mock_sp.EncodeAsIds.assert_called_once_with(multilingual_text)

    def test_unexpected_input_types(self):
        # Test with various unexpected input types
        unexpected_inputs = [None, 123, 3.14, [], {}, set()]
        for input_value in unexpected_inputs:
            with self.assertRaises(ValueError):
                self.tokenizer.encode(input_value)

    def test_large_token_list(self):
        # Test decoding a large list of tokens
        large_token_list = list(range(1000000))  # 1 million tokens
        self.mock_sp.DecodeIds.return_value = "a" * 1000000  # Simulate decoding to a very long string
        decoded_text = self.tokenizer.decode(large_token_list)
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(len(decoded_text), 1000000)
        # Check if DecodeIds was called with the correct arguments
        expected_tokens = [token for token in large_token_list if token not in {self.tokenizer.bos_id, self.tokenizer.eos_id, self.tokenizer.pad_id}]
        self.mock_sp.DecodeIds.assert_called_once_with(expected_tokens)
        # Verify that the mock was actually called
        self.assertTrue(self.mock_sp.DecodeIds.called, "DecodeIds was not called")

if __name__ == '__main__':
    unittest.main()
