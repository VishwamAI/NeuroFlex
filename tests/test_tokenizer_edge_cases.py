import unittest
from unittest.mock import Mock, patch
from NeuroFlex.utils.tokenizer import Tokenizer
from transformers import AutoTokenizer

class TestTokenizerEdgeCases(unittest.TestCase):
    def setUp(self):
        # Mock AutoTokenizer
        self.mock_auto_tokenizer = Mock()
        self.mock_auto_tokenizer.from_pretrained.return_value = Mock()

        # Patch AutoTokenizer in Tokenizer
        self.mock_auto_tokenizer_patcher = patch('NeuroFlex.utils.tokenizer.AutoTokenizer', self.mock_auto_tokenizer)
        self.mock_auto_tokenizer_patcher.start()

        self.tokenizer = Tokenizer()

    def tearDown(self):
        # Stop all patches
        self.mock_auto_tokenizer_patcher.stop()

    def test_very_long_text(self):
        long_text = "a" * 1_000_000
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.return_value = list(range(100000))  # Simulate 100,000 tokens
        tokens = self.tokenizer.encode(long_text)
        self.assertEqual(len(tokens), 100000)
        mock_encode.assert_called_once_with(long_text, add_special_tokens=True)

    def test_text_with_special_characters(self):
        special_char_text = "!@#$%^&*()_+{}[]|\\:;\"'<>,.?/~`"
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.return_value = list(range(10, 40))  # Simulate 30 tokens
        tokens = self.tokenizer.encode(special_char_text)
        self.assertEqual(len(tokens), 30)
        mock_encode.assert_called_once_with(special_char_text, add_special_tokens=True)

    def test_multilingual_text(self):
        multilingual_text = "Hello こんにちは Bonjour नमस्ते 你好"
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.return_value = list(range(10, 20))  # Simulate 10 tokens
        tokens = self.tokenizer.encode(multilingual_text)
        self.assertEqual(len(tokens), 10)
        mock_encode.assert_called_once_with(multilingual_text, add_special_tokens=True)

    def test_unexpected_input_types(self):
        unexpected_inputs = [None, 123, 3.14, [], {}, set()]
        for input_value in unexpected_inputs:
            with self.assertRaises(ValueError):
                self.tokenizer.encode(input_value)

    def test_large_token_list(self):
        large_token_list = list(range(1000000))  # 1 million tokens
        mock_decode = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        mock_decode.return_value = "a" * 1000000  # Simulate decoding to a very long string
        decoded_text = self.tokenizer.decode(large_token_list)
        self.assertIsInstance(decoded_text, str)
        self.assertEqual(len(decoded_text), 1000000)
        mock_decode.assert_called_once_with(large_token_list, skip_special_tokens=True)

    def test_empty_input(self):
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.return_value = []
        tokens = self.tokenizer.encode("")
        self.assertEqual(len(tokens), 0)
        mock_encode.assert_called_once_with("", add_special_tokens=True)

    def test_decode_empty_list(self):
        decoded_text = self.tokenizer.decode([])
        self.assertEqual(decoded_text, "")

    def test_decode_none(self):
        with self.assertRaises(ValueError):
            self.tokenizer.decode(None)

    @patch('NeuroFlex.utils.tokenizer.logger.error')
    def test_encode_error_logging(self, mock_log_error):
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.side_effect = Exception("Encoding error")
        with self.assertRaises(Exception):
            self.tokenizer.encode("Test")
        mock_log_error.assert_called_once_with("Error encoding text: Encoding error")

    @patch('NeuroFlex.utils.tokenizer.logger.error')
    def test_decode_error_logging(self, mock_log_error):
        mock_decode = self.mock_auto_tokenizer.from_pretrained.return_value.decode
        mock_decode.side_effect = Exception("Decoding error")
        with self.assertRaises(Exception):
            self.tokenizer.decode([1, 2, 3])
        mock_log_error.assert_called_once_with("Error decoding token ids: Decoding error")

    def test_load_pretrained_tokenizers(self):
        self.tokenizer.load_pretrained_tokenizers("bert-base-multilingual-cased")
        self.mock_auto_tokenizer.from_pretrained.assert_called_with("bert-base-multilingual-cased")

    def test_add_special_tokens(self):
        mock_tokenizer = self.mock_auto_tokenizer.from_pretrained.return_value
        mock_tokenizer.special_tokens_map = {'unk_token': '[UNK]'}
        mock_tokenizer.add_special_tokens.return_value = 2

        special_tokens = ["[NEW1]", "[NEW2]"]
        self.tokenizer.add_special_tokens(special_tokens)

        mock_tokenizer.add_special_tokens.assert_called_once_with({'additional_special_tokens': special_tokens})

    def test_token_to_id(self):
        mock_tokenizer = self.mock_auto_tokenizer.from_pretrained.return_value
        mock_tokenizer.convert_tokens_to_ids.return_value = 42

        token_id = self.tokenizer.token_to_id("test_token")
        self.assertEqual(token_id, 42)
        mock_tokenizer.convert_tokens_to_ids.assert_called_once_with("test_token")

    def test_id_to_token(self):
        mock_tokenizer = self.mock_auto_tokenizer.from_pretrained.return_value
        mock_tokenizer.convert_ids_to_tokens.return_value = "test_token"

        token = self.tokenizer.id_to_token(42)
        self.assertEqual(token, "test_token")
        mock_tokenizer.convert_ids_to_tokens.assert_called_once_with(42)

    def test_non_ascii_characters(self):
        non_ascii_text = "Café au lait • ¡Hola! • こんにちは • 你好 • Здравствуйте"
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        mock_encode.return_value = list(range(15))  # Simulate 15 tokens
        tokens = self.tokenizer.encode(non_ascii_text)
        self.assertEqual(len(tokens), 15)
        mock_encode.assert_called_once_with(non_ascii_text, add_special_tokens=True)

if __name__ == '__main__':
    unittest.main()
