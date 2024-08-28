import unittest
from unittest.mock import Mock, patch
import os
from NeuroFlex.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
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

    def test_encode_basic(self):
        text = "Hello, world!"
        self.mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertEqual(len(tokens), 5)  # BOS + 3 tokens + EOS
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.assertEqual(tokens[1:-1], [10, 20, 30])
        self.mock_sp.EncodeAsIds.assert_called_once_with(text)

    def test_encode_no_bos(self):
        text = "Hello, world!"
        self.mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        tokens = self.tokenizer.encode(text, bos=False)
        self.assertEqual(len(tokens), 4)  # 3 tokens + EOS
        self.assertNotEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.assertEqual(tokens[:-1], [10, 20, 30])
        self.mock_sp.EncodeAsIds.assert_called_once_with(text)

    def test_encode_no_eos(self):
        text = "Hello, world!"
        self.mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        tokens = self.tokenizer.encode(text, eos=False)
        self.assertEqual(len(tokens), 4)  # BOS + 3 tokens
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertNotEqual(tokens[-1], self.tokenizer.eos_id)
        self.assertEqual(tokens[1:], [10, 20, 30])
        self.mock_sp.EncodeAsIds.assert_called_once_with(text)

    def test_decode_basic(self):
        text = "Hello, world!"
        encoded_tokens = [10, 20, 30]
        tokens = [self.tokenizer.bos_id] + encoded_tokens + [self.tokenizer.eos_id]
        self.mock_sp.DecodeIds.return_value = text
        decoded_text = self.tokenizer.decode(tokens)
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(decoded_text, text)
        self.mock_sp.DecodeIds.assert_called_once_with(encoded_tokens)

    def test_encode_decode_roundtrip(self):
        original_text = "This is a test sentence."
        encoded_tokens = [10, 20, 30, 40, 50]
        self.mock_sp.EncodeAsIds.return_value = encoded_tokens
        tokens = self.tokenizer.encode(original_text)
        self.mock_sp.DecodeIds.return_value = original_text
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(original_text, decoded_text)
        self.mock_sp.EncodeAsIds.assert_called_once_with(original_text)
        self.mock_sp.DecodeIds.assert_called_once_with(encoded_tokens)
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.assertEqual(tokens[1:-1], encoded_tokens)
        self.assertEqual(len(tokens), len(encoded_tokens) + 2)  # +2 for BOS and EOS tokens

    def test_empty_string(self):
        text = ""
        self.mock_sp.EncodeAsIds.return_value = []
        tokens = self.tokenizer.encode(text)
        self.assertEqual(len(tokens), 2)  # Should contain BOS and EOS tokens
        self.assertEqual(tokens[0], self.tokenizer.bos_id)
        self.assertEqual(tokens[-1], self.tokenizer.eos_id)
        self.mock_sp.DecodeIds.return_value = ""
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, "[DECODING_FAILED]")
        # DecodeIds should be called with an empty list
        self.mock_sp.DecodeIds.assert_called_once_with([])

    def test_special_tokens_handling(self):
        text = "Special tokens test"
        self.mock_sp.EncodeAsIds.return_value = [10, 20, 30]
        tokens = self.tokenizer.encode(text)
        expected_tokens = [self.tokenizer.bos_id, 10, 20, 30, self.tokenizer.eos_id]
        self.assertEqual(tokens, expected_tokens)
        self.mock_sp.EncodeAsIds.assert_called_once_with(text)

        self.mock_sp.DecodeIds.return_value = text
        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, text)
        self.mock_sp.DecodeIds.assert_called_once_with([10, 20, 30])

        # Test handling of special tokens during decoding
        special_tokens = [self.tokenizer.bos_id, 10, 20, 30, self.tokenizer.eos_id]
        self.mock_sp.DecodeIds.return_value = text
        decoded_special = self.tokenizer.decode(special_tokens)
        self.assertEqual(decoded_special, text)
        self.mock_sp.DecodeIds.assert_called_with([10, 20, 30])

    def test_special_tokens(self):
        self.assertIsNotNone(self.tokenizer.bos_id)
        self.assertIsNotNone(self.tokenizer.eos_id)
        self.assertIsNotNone(self.tokenizer.pad_id)
        self.assertIsNotNone(self.tokenizer.unk_id)
        self.assertIsNotNone(self.tokenizer.space_id)
        self.assertNotEqual(self.tokenizer.bos_id, self.tokenizer.eos_id)
        self.assertNotEqual(self.tokenizer.bos_id, self.tokenizer.pad_id)
        self.assertNotEqual(self.tokenizer.eos_id, self.tokenizer.pad_id)

    def test_encode_invalid_input(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode(123)  # Non-string input

    def test_decode_invalid_input(self):
        # Test empty list
        print("Testing empty list")
        result = self.tokenizer.decode([])
        print(f"Result: {result}")
        self.assertEqual(result, "[DECODING_FAILED]")
        self.mock_sp.DecodeIds.assert_called_once_with([])
        self.mock_sp.DecodeIds.reset_mock()

        # Test list with non-integer values
        print("Testing list with non-integer values")
        result = self.tokenizer.decode(["a", "b", "c"])
        print(f"Result: {result}")
        self.assertEqual(result, "[DECODING_FAILED]")
        self.mock_sp.DecodeIds.assert_not_called()
        self.mock_sp.DecodeIds.reset_mock()

        # Test with None
        print("Testing None input")
        result = self.tokenizer.decode(None)
        print(f"Result: {result}")
        self.assertEqual(result, "[DECODING_FAILED]")
        self.mock_sp.DecodeIds.assert_not_called()
        self.mock_sp.DecodeIds.reset_mock()

        # Test with invalid integer values
        print("Testing invalid integer values")
        self.mock_sp.DecodeIds.side_effect = Exception("Invalid token ID")
        result = self.tokenizer.decode([-1, 1000000])
        print(f"Result: {result}")
        self.assertEqual(result, "[DECODING_FAILED]")
        self.mock_sp.DecodeIds.assert_called_once_with([-1, 1000000])
        self.mock_sp.DecodeIds.reset_mock()

        # Reset side_effect for subsequent tests
        self.mock_sp.DecodeIds.side_effect = None

    def test_decode_error_handling(self):
        self.mock_sp.DecodeIds.side_effect = Exception("Decoding error")
        result = self.tokenizer.decode([10, 20, 30])
        self.assertEqual(result, "[DECODING_FAILED]")

    @patch('logging.error')
    def test_logging_on_error(self, mock_log_error):
        self.mock_sp.EncodeAsIds.side_effect = Exception("Encoding error")
        with self.assertRaises(ValueError):
            self.tokenizer.encode("Test")
        mock_log_error.assert_called()

    def test_tokenize_detokenize(self):
        text = "This is a test sentence."
        self.mock_sp.EncodeAsPieces.return_value = ['▁This', '▁is', '▁a', '▁test', '▁sentence', '.']
        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, ['This', 'is', 'a', 'test', 'sentence', '.'])

        # Mock the _post_process_decoded_text method
        with patch.object(self.tokenizer, '_post_process_decoded_text', return_value=text):
            detokenized = self.tokenizer.detokenize(tokens)
            self.assertEqual(detokenized, text)

        self.mock_sp.EncodeAsPieces.assert_called_once_with(text)
        self.mock_sp.DecodePieces.assert_called_once_with(tokens)

        # Test with tokens containing leading underscores
        tokens_with_underscores = ['▁This', '▁is', '▁a', '▁test', '▁sentence', '.']
        expected_tokens = ['This', 'is', 'a', 'test', 'sentence', '.']
        self.assertEqual(self.tokenizer.tokenize(text), expected_tokens)

if __name__ == '__main__':
    unittest.main()
