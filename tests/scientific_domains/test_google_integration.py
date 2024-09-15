import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import tensorflow as tf
from NeuroFlex.scientific_domains.google_integration import GoogleIntegration

class TestGoogleIntegration(unittest.TestCase):
    def setUp(self):
        self.input_shape = (28, 28, 1)
        self.num_classes = 10
        self.google_integration = GoogleIntegration(self.input_shape, self.num_classes)

    def test_create_cnn_model(self):
        cnn_model = self.google_integration.create_cnn_model()
        self.assertIsInstance(cnn_model, nn.Module)

        # Test forward pass
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1,) + self.input_shape)
        params = cnn_model.init(key, x)
        output = cnn_model.apply(params, x)
        self.assertEqual(output.shape, (1, self.num_classes))





    def test_xla_compilation(self):
        cnn_model = self.google_integration.create_cnn_model()
        compiled_fn = self.google_integration.xla_compilation(cnn_model, (1,) + self.input_shape)
        self.assertTrue(callable(compiled_fn))

        # Test compiled function
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1,) + self.input_shape)
        params = cnn_model.init(key, x)
        output = compiled_fn(params, x)
        self.assertEqual(output.shape, (1, self.num_classes))

    def test_integrate_tensorflow_model(self):
        # Create a simple TensorFlow model
        tf_model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=self.input_shape),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.num_classes)
        ])

        integrated_model = self.google_integration.integrate_tensorflow_model(tf_model)
        self.assertIsInstance(integrated_model, nn.Module)

        # Test forward pass
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (1,) + self.input_shape)
        params = integrated_model.init(key, x)
        output = integrated_model.apply(params, x)
        self.assertEqual(output.shape, (1, self.num_classes))

    def test_input_shape_handling(self):
        # Test with different input shapes
        test_shapes = [(32, 32, 3), (64, 64, 1), (224, 224, 3)]
        for shape in test_shapes:
            gi = GoogleIntegration(shape, self.num_classes)
            cnn_model = gi.create_cnn_model()
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (1,) + shape)
            params = cnn_model.init(key, x)
            output = cnn_model.apply(params, x)
            self.assertEqual(output.shape, (1, self.num_classes))

    def test_num_classes_handling(self):
        # Test with different number of classes
        test_classes = [2, 5, 100]
        for num_classes in test_classes:
            gi = GoogleIntegration(self.input_shape, num_classes)
            cnn_model = gi.create_cnn_model()
            key = jax.random.PRNGKey(0)
            x = jax.random.normal(key, (1,) + self.input_shape)
            params = cnn_model.init(key, x)
            output = cnn_model.apply(params, x)
            self.assertEqual(output.shape, (1, num_classes))

    def test_error_handling(self):
        # Test with invalid input shape
        with self.assertRaises(ValueError):
            GoogleIntegration((28, 28), self.num_classes)  # Missing channel dimension

        # Test with invalid number of classes
        with self.assertRaises(ValueError):
            GoogleIntegration(self.input_shape, 0)  # Number of classes should be positive

if __name__ == '__main__':
    unittest.main()
