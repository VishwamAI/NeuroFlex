import unittest
import time
import sys
from unittest.mock import Mock, patch
import os
from memory_profiler import profile
from NeuroFlex.tokenizer import Tokenizer

class TestTokenizerPerformance(unittest.TestCase):
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

    def generate_large_text(self, size):
        return "This is a test sentence. " * size

    @profile
    def test_encode_performance(self):
        text_sizes = [1000, 10000, 100000]
        for size in text_sizes:
            large_text = self.generate_large_text(size)
            self.mock_sp.EncodeAsIds.return_value = list(range(size))

            start_time = time.time()
            tokens = self.tokenizer.encode(large_text)
            end_time = time.time()

            encoding_time = end_time - start_time
            print(f"Encoding time for {size} words: {encoding_time:.4f} seconds")

            self.assertIsInstance(tokens, list)
            self.assertGreater(len(tokens), size)

    @profile
    def test_decode_performance(self):
        token_sizes = [1000, 10000, 100000]
        for size in token_sizes:
            tokens = list(range(size))
            self.mock_sp.DecodeIds.return_value = "a" * size

            start_time = time.time()
            decoded_text = self.tokenizer.decode(tokens)
            end_time = time.time()

            decoding_time = end_time - start_time
            print(f"Decoding time for {size} tokens: {decoding_time:.4f} seconds")

            self.assertIsInstance(decoded_text, str)
            self.assertGreater(len(decoded_text), 0)

if __name__ == '__main__':
    unittest.main()
