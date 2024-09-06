import unittest
from NeuroFlex.utils.tokenizer import Tokenizer
from transformers import AutoTokenizer

class TestTokenisation(unittest.TestCase):
    def setUp(self):
        self.tokenizer = Tokenizer()

    def test_basic_tokenization(self):
        text = "Hello, world! This is a test."
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_empty_string(self):
        self.assertEqual(self.tokenizer.tokenize(""), [])

    def test_special_characters(self):
        text = "!@#$%^&*()_+"
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_numbers(self):
        text = "I have 3 apples and 2.5 oranges."
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_contractions(self):
        text = "I'm can't won't"
        tokens = self.tokenizer.tokenize(text)
        self.assertIsInstance(tokens, list)
        self.assertTrue(all(isinstance(token, str) for token in tokens))

    def test_encode_decode(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        decoded = self.tokenizer.decode(encoded)
        self.assertEqual(decoded, text)

    def test_tokenizer_initialization(self):
        custom_tokenizer = Tokenizer("bert-base-uncased")
        self.assertIsInstance(custom_tokenizer.tokenizers['default'], AutoTokenizer)

    def test_encode(self):
        text = "Hello, world!"
        encoded = self.tokenizer.encode(text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(token, int) for token in encoded))

    def test_get_vocab(self):
        vocab = self.tokenizer.get_vocab()
        self.assertIsInstance(vocab, dict)
        self.assertTrue(all(isinstance(token, str) and isinstance(id, int) for token, id in vocab.items()))

    def test_get_vocab_size(self):
        vocab_size = self.tokenizer.get_vocab_size()
        self.assertIsInstance(vocab_size, int)
        self.assertGreater(vocab_size, 0)

    def test_load_pretrained_tokenizers(self):
        self.tokenizer.load_pretrained_tokenizers("bert-base-multilingual-cased")
        self.assertIsInstance(self.tokenizer.tokenizer, AutoTokenizer)

    def test_add_special_tokens(self):
        special_tokens = ["[NEW1]", "[NEW2]"]
        original_vocab_size = self.tokenizer.get_vocab_size()
        self.tokenizer.add_special_tokens(special_tokens)
        new_vocab_size = self.tokenizer.get_vocab_size()
        self.assertGreater(new_vocab_size, original_vocab_size)

    def test_token_to_id_and_id_to_token(self):
        token = "hello"
        token_id = self.tokenizer.token_to_id(token)
        self.assertIsInstance(token_id, int)
        recovered_token = self.tokenizer.id_to_token(token_id)
        self.assertEqual(token, recovered_token)

if __name__ == '__main__':
    unittest.main()
