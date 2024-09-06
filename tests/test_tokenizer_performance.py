import unittest
import time
import psutil
from unittest.mock import Mock, patch
from memory_profiler import profile
from NeuroFlex.utils.tokenizer import Tokenizer

# Constants
NUM_ITERATIONS = 100000
LARGE_TEXT_SIZES = [10000, 100000, 1000000]

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert to MB

class TestTokenizerPerformance(unittest.TestCase):
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

    def generate_large_text(self, size):
        return "This is a test sentence. " * size

    def run_performance_test(self, operation, sizes, setup_func, test_func):
        for size in sizes:
            input_data = setup_func(size)
            mock_func = getattr(self.mock_auto_tokenizer.from_pretrained.return_value, operation)
            mock_func.return_value = input_data if operation == 'decode' else list(range(len(input_data)))

            start_time = time.time()
            start_memory = get_memory_usage()

            try:
                result = test_func(input_data)
            except Exception as e:
                self.fail(f"Error during {operation} operation: {str(e)}")

            end_time = time.time()
            end_memory = get_memory_usage()

            operation_time = end_time - start_time
            memory_used = end_memory - start_memory

            print(f"{operation.capitalize()} performance for size {size}:")
            print(f"  Time: {operation_time:.4f} seconds")
            print(f"  Memory: {memory_used:.2f} MB")
            print(f"  Throughput: {size / operation_time:.2f} items/second")

            self.assertIsNotNone(result)
            mock_func.assert_called_once()

            # Reset mock for next iteration
            mock_func.reset_mock()

    @profile
    def test_encode_performance(self):
        try:
            self.run_performance_test(
                'encode',
                [10000, 100000, 1000000],
                self.generate_large_text,
                self.tokenizer.encode
            )
        except Exception as e:
            self.fail(f"Encode performance test failed: {str(e)}")

    @profile
    def test_decode_performance(self):
        try:
            self.run_performance_test(
                'decode',
                [10000, 100000, 1000000],
                lambda size: list(range(size)),
                self.tokenizer.decode
            )
        except Exception as e:
            self.fail(f"Decode performance test failed: {str(e)}")

    @profile
    def test_tokenize_performance(self):
        try:
            self.run_performance_test(
                'tokenize',
                [10000, 100000, 1000000],
                self.generate_large_text,
                self.tokenizer.tokenize
            )
        except Exception as e:
            self.fail(f"Tokenize performance test failed: {str(e)}")

        # Test with non-ASCII text
        non_ascii_text = "こんにちは世界" * 10000  # "Hello World" in Japanese, repeated
        self.run_performance_test(
            'tokenize',
            [len(non_ascii_text)],
            lambda x: non_ascii_text,
            self.tokenizer.tokenize
        )

    @profile
    def test_token_to_id_performance(self):
        def setup_func(size):
            return ["test_token"] * size

        def test_func(tokens):
            return [self.tokenizer.token_to_id(token) for token in tokens]

        mock_convert = self.mock_auto_tokenizer.from_pretrained.return_value.convert_tokens_to_ids
        mock_convert.return_value = 42

        self.run_performance_test(
            'convert_tokens_to_ids',
            [10000, 100000, 1000000],
            setup_func,
            test_func
        )

        self.assertEqual(test_func(["test_token"])[0], 42)
        mock_convert.assert_called_with("test_token")

    @profile
    def test_id_to_token_performance(self):
        def setup_func(size):
            return [i for i in range(size)]

        def test_func(input_data):
            return [self.tokenizer.id_to_token(token_id) for token_id in input_data]

        self.run_performance_test(
            'convert_ids_to_tokens',
            [1000, 10000, 100000],
            setup_func,
            test_func
        )

        # Additional test for a single conversion
        token_id = 42
        mock_convert = self.mock_auto_tokenizer.from_pretrained.return_value.convert_ids_to_tokens
        mock_convert.return_value = "test_token"

        token = self.tokenizer.id_to_token(token_id)
        self.assertEqual(token, "test_token")
        mock_convert.assert_called_with(token_id)

    @profile
    def test_non_ascii_performance(self):
        non_ascii_text = "Café au lait • ¡Hola! • こんにちは • 你好 • Здравствуйте • 🌍🚀🎉 • ñáéíóú • αβγδε • ℕ∈ℝ∀∃∇"
        sizes = [1000, 10000, 100000]

        def setup_func(size):
            return non_ascii_text * size

        def test_func(input_data):
            return self.tokenizer.encode(input_data)

        self.run_performance_test('encode', sizes, setup_func, test_func)

        # Additional assertions
        mock_encode = self.mock_auto_tokenizer.from_pretrained.return_value.encode
        tokens = test_func(setup_func(sizes[0]))
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
        mock_encode.assert_called_with(setup_func(sizes[0]), add_special_tokens=True)

        # Test decoding performance for non-ASCII text
        def decode_test_func(input_data):
            return self.tokenizer.decode(input_data)

        self.run_performance_test('decode', sizes, lambda size: list(range(size)), decode_test_func)

        # Test tokenization performance for non-ASCII text
        def tokenize_test_func(input_data):
            return self.tokenizer.tokenize(input_data)

        self.run_performance_test('tokenize', sizes, setup_func, tokenize_test_func)

if __name__ == '__main__':
    unittest.main()
