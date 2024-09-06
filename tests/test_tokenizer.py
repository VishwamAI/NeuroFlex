import unittest
from unittest.mock import Mock, patch
from NeuroFlex.utils.tokenizer import Tokenizer

class TestTokenizer(unittest.TestCase):
    def setUp(self):
        # Mock AutoTokenizer
        self.mock_auto_tokenizer = Mock()
        self.mock_auto_tokenizer.from_pretrained.return_value = Mock()

        # Patch AutoTokenizer in Tokenizer
        self.mock_auto_tokenizer_patcher = patch('NeuroFlex.utils.tokenizer.AutoTokenizer', self.mock_auto_tokenizer)
        self.mock_auto_tokenizer_patcher.start()

        # Initialize Tokenizer with default parameters
        self.tokenizer = Tokenizer()

        # Set up special tokens
        self.special_tokens = {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3}
        self.tokenizer.special_tokens = self.special_tokens

    def tearDown(self):
        # Stop all patches
        self.mock_auto_tokenizer_patcher.stop()

    def test_encode_basic(self):
        text = "Hello, world!"
        mock_encoded = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encoded.return_value = [101, 7592, 1010, 2088, 999, 102]  # Example BERT encoding
        tokens = self.tokenizer.encode(text)
        self.assertIsInstance(tokens, list)
        self.assertEqual(tokens, [101, 7592, 1010, 2088, 999, 102])
        mock_encoded.assert_called_once_with(text, add_special_tokens=True)

    def test_decode_basic(self):
        token_ids = [101, 7592, 1010, 2088, 999, 102]
        expected_text = "Hello, world!"
        mock_decoded = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        mock_decoded.return_value = expected_text
        decoded_text = self.tokenizer.decode(token_ids)
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(decoded_text, expected_text)
        mock_decoded.assert_called_once_with(token_ids, skip_special_tokens=True)

    def test_encode_decode_roundtrip(self):
        original_text = "This is a test sentence."
        mock_encoded = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_decoded = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        encoded_tokens = [101, 2023, 2003, 1037, 3231, 6251, 1012, 102]
        mock_encoded.return_value = encoded_tokens
        mock_decoded.return_value = original_text

        tokens = self.tokenizer.encode(original_text)
        self.assertEqual(tokens, encoded_tokens)

        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, original_text)

        mock_encoded.assert_called_once_with(original_text, add_special_tokens=True)
        mock_decoded.assert_called_once_with(encoded_tokens, skip_special_tokens=True)

    def test_empty_string(self):
        text = ""
        mock_encoded = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_decoded = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        mock_encoded.return_value = [101, 102]  # CLS and SEP tokens
        mock_decoded.return_value = ""

        tokens = self.tokenizer.encode(text)
        self.assertEqual(tokens, [101, 102])

        decoded_text = self.tokenizer.decode(tokens)
        self.assertEqual(decoded_text, "")

        mock_encoded.assert_called_once_with(text, add_special_tokens=True)
        mock_decoded.assert_called_once_with([101, 102], skip_special_tokens=True)

    def test_special_tokens(self):
        self.assertEqual(self.tokenizer.special_tokens['<unk>'], 0)
        self.assertEqual(self.tokenizer.special_tokens['<s>'], 1)
        self.assertEqual(self.tokenizer.special_tokens['</s>'], 2)
        self.assertEqual(self.tokenizer.special_tokens['<pad>'], 3)

    def test_encode_invalid_input(self):
        with self.assertRaises(ValueError):
            self.tokenizer.encode(123)  # Non-string input

    def test_decode_invalid_input(self):
        mock_decoded = self.mock_auto_tokenizer.from_pretrained.return_value.decode

        # Test empty list
        result = self.tokenizer.decode([])
        self.assertEqual(result, "")
        mock_decoded.assert_not_called()

        # Test None
        with self.assertRaises(ValueError):
            self.tokenizer.decode(None)
        mock_decoded.assert_not_called()

        # Test list with non-integer values
        with self.assertRaises(ValueError):
            self.tokenizer.decode(["a", "b", "c"])
        mock_decoded.assert_not_called()

        # Test with invalid integer values
        mock_decoded.side_effect = Exception("Invalid input")
        with self.assertRaises(Exception):
            self.tokenizer.decode([-1, 1000000])
        mock_decoded.assert_called_once_with([-1, 1000000], skip_special_tokens=True)
        mock_decoded.reset_mock()

        # Test with mixed valid and invalid integer values
        mock_decoded.side_effect = None
        mock_decoded.return_value = "Valid token"
        result = self.tokenizer.decode([1, -1, 1000000, 2])
        self.assertEqual(result, "Valid token")
        mock_decoded.assert_called_once_with([1, -1, 1000000, 2], skip_special_tokens=True)

        # Reset side_effect for subsequent tests
        mock_decoded.side_effect = None

    @patch('NeuroFlex.utils.tokenizer.logger.error')
    def test_logging_on_error(self, mock_log_error):
        # Test encode error logging
        mock_encoded = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encoded.side_effect = Exception("Encoding error")

        with self.assertRaises(Exception) as context:
            self.tokenizer.encode("Test")

        self.assertEqual(str(context.exception), "Encoding error")
        mock_log_error.assert_called_once_with("Error encoding text: Encoding error")

        # Test decode error logging
        mock_decoded = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        mock_decoded.side_effect = Exception("Decoding error")

        with self.assertRaises(Exception) as context:
            self.tokenizer.decode([1, 2, 3])

        self.assertEqual(str(context.exception), "Decoding error")
        mock_log_error.assert_called_with("Error decoding token ids: Decoding error")

        # Verify that both encode and decode errors are logged
        self.assertEqual(mock_log_error.call_count, 2)
        mock_log_error.assert_any_call("Error encoding text: Encoding error")
        mock_log_error.assert_any_call("Error decoding token ids: Decoding error")

        # Reset mock and test both errors again to ensure consistent behavior
        mock_log_error.reset_mock()
        mock_encoded.side_effect = Exception("Another encoding error")
        mock_decoded.side_effect = Exception("Another decoding error")

        with self.assertRaises(Exception):
            self.tokenizer.encode("Another test")

        with self.assertRaises(Exception):
            self.tokenizer.decode([4, 5, 6])

        self.assertEqual(mock_log_error.call_count, 2)
        mock_log_error.assert_any_call("Error encoding text: Another encoding error")
        mock_log_error.assert_any_call("Error decoding token ids: Another decoding error")

    def test_tokenize(self):
        text = "This is a test sentence."
        mock_tokenize = self.mock_auto_tokenizer.from_pretrained.return_value.tokenize
        expected_tokens = ['this', 'is', 'a', 'test', 'sentence', '.']
        mock_tokenize.return_value = expected_tokens

        tokens = self.tokenizer.tokenize(text)
        self.assertEqual(tokens, expected_tokens)
        mock_tokenize.assert_called_once_with(text)

    def test_get_vocab(self):
        mock_vocab = {'test': 0, 'vocab': 1}
        mock_get_vocab = self.mock_auto_tokenizer.from_pretrained.return_value.get_vocab
        mock_get_vocab.return_value = mock_vocab

        vocab = self.tokenizer.get_vocab()
        self.assertEqual(vocab, mock_vocab)
        mock_get_vocab.assert_called_once()

    def test_get_vocab_size(self):
        mock_vocab = {'test': 0, 'vocab': 1}
        mock_get_vocab = self.mock_auto_tokenizer.from_pretrained.return_value.get_vocab
        mock_get_vocab.return_value = mock_vocab

        vocab_size = self.tokenizer.get_vocab_size()
        self.assertEqual(vocab_size, 2)
        mock_get_vocab.assert_called_once()

    def test_token_to_id(self):
        mock_convert = self.mock_auto_tokenizer.from_pretrained.return_value.convert_tokens_to_ids
        mock_convert.return_value = 42

        token_id = self.tokenizer.token_to_id("test")
        self.assertEqual(token_id, 42)
        mock_convert.assert_called_once_with("test")

    def test_id_to_token(self):
        mock_convert = self.mock_auto_tokenizer.from_pretrained.return_value.convert_ids_to_tokens
        mock_convert.return_value = "test"

        token = self.tokenizer.id_to_token(42)
        self.assertEqual(token, "test")
        mock_convert.assert_called_once_with(42)

if __name__ == '__main__':
    unittest.main()
