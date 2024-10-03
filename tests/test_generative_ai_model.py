import unittest
import torch
from NeuroFlex.generative_models.generative_ai_model import GenerativeAIModel, load_model

class TestGenerativeAIModel(unittest.TestCase):
    def setUp(self):
        self.model = GenerativeAIModel(sp_model_path="path/to/sp_model.model")

    def test_model_initialization(self):
        self.assertIsInstance(self.model, GenerativeAIModel)
        self.assertIsNotNone(self.model.tokenizer)
        self.assertIsNotNone(self.model.model)
        self.assertIsNotNone(self.model.sp_processor)

    def test_forward_method(self):
        input_text = "Translate to French: Hello, how are you?"
        output = self.model.forward(input_text)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        self.assertIsInstance(output[0], str)

    def test_generate_method(self):
        input_text = "Summarize: The quick brown fox jumps over the lazy dog."
        output = self.model.generate(input_text, max_length=30)
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)
        self.assertIsInstance(output[0], str)
        self.assertLessEqual(len(self.model.sp_processor.encode(output[0])), 30)

    def test_load_model(self):
        loaded_model = load_model(sp_model_path="path/to/sp_model.model")
        self.assertIsInstance(loaded_model, GenerativeAIModel)
        self.assertIsNotNone(loaded_model.sp_processor)

    def test_model_output_consistency(self):
        input_text = "Translate to Spanish: Good morning!"
        output1 = self.model.generate(input_text)
        output2 = self.model.generate(input_text)
        self.assertEqual(output1, output2, "Model output should be consistent for the same input")

    def test_sentencepiece_integration(self):
        input_text = "This is a test sentence for SentencePiece."
        sp_tokens = self.model.sp_processor.encode(input_text, out_type=str)
        self.assertIsInstance(sp_tokens, list)
        self.assertGreater(len(sp_tokens), 0)
        decoded_text = self.model.sp_processor.decode(sp_tokens)
        self.assertEqual(input_text, decoded_text)

if __name__ == '__main__':
    unittest.main()
