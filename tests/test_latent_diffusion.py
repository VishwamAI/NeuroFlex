import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn

from NeuroFlex.generative_models.latent_diffusion import LatentDiffusionModel

class TestLatentDiffusion(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.latent_dim = 64
        self.image_size = (32, 32, 3)
        self.model = LatentDiffusionModel(latent_dim=self.latent_dim, image_size=self.image_size)

    def test_model_initialization(self):
        self.assertIsInstance(self.model, LatentDiffusionModel)
        self.assertEqual(self.model.latent_dim, self.latent_dim)
        self.assertEqual(self.model.image_size, self.image_size)

    def test_forward_pass(self):
        input_shape = (1, *self.image_size)
        dummy_input = jax.random.normal(self.rng, input_shape)
        output = self.model(dummy_input)
        self.assertEqual(output.shape, input_shape)

    def test_train_step(self):
        input_shape = (32, *self.image_size)
        dummy_input = jax.random.normal(self.rng, input_shape)
        dummy_target = jax.random.normal(self.rng, input_shape)

        loss = self.model.compute_loss(dummy_input, dummy_target)

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())

    def test_generate(self):
        num_samples = 1
        generated = self.model.generate(self.rng, num_samples)

        self.assertEqual(generated.shape, (num_samples, *self.image_size))

    def test_encode_decode(self):
        input_shape = (1, *self.image_size)
        dummy_input = jax.random.normal(self.rng, input_shape)

        latent = self.model.encode(dummy_input)
        reconstructed = self.model.decode(latent)

        self.assertEqual(latent.shape, (1, self.latent_dim))
        self.assertEqual(reconstructed.shape, input_shape)

if __name__ == '__main__':
    unittest.main()
