import unittest
import jax.numpy as jnp
from NeuroFlex.scientific_domains import GoogleIntegration

class TestGoogleIntegration(unittest.TestCase):
    def setUp(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.google_integration = GoogleIntegration(self.input_shape, self.num_classes)

    def test_create_cnn_model(self):
        cnn_model = self.google_integration.create_cnn_model()
        self.assertIsNotNone(cnn_model)
        self.assertTrue(hasattr(cnn_model, '__call__'))

    def test_create_rnn_model(self):
        rnn_model = self.google_integration.create_rnn_model()
        self.assertIsNotNone(rnn_model)
        self.assertTrue(hasattr(rnn_model, '__call__'))

    def test_create_transformer_model(self):
        transformer_model = self.google_integration.create_transformer_model()
        self.assertIsNotNone(transformer_model)
        self.assertTrue(hasattr(transformer_model, '__call__'))

    def test_xla_compilation(self):
        cnn_model = self.google_integration.create_cnn_model()
        compiled_cnn = self.google_integration.xla_compilation(cnn_model, (1,) + self.input_shape)
        self.assertIsNotNone(compiled_cnn)
        self.assertTrue(callable(compiled_cnn))

if __name__ == '__main__':
    unittest.main()
