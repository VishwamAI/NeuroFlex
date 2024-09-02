import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.gan import GAN

class TestGAN(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.latent_dim = 100
        self.generator_features = (128, 256, 512)
        self.discriminator_features = (512, 256, 128)
        self.output_shape = (28, 28, 1)
        self.batch_size = 32

    def test_gan_initialization(self):
        gan = GAN(
            latent_dim=self.latent_dim,
            generator_features=self.generator_features,
            discriminator_features=self.discriminator_features,
            output_shape=self.output_shape
        )
        self.assertIsInstance(gan, GAN)
        self.assertEqual(gan.latent_dim, self.latent_dim)
        self.assertEqual(gan.generator_features, self.generator_features)
        self.assertEqual(gan.discriminator_features, self.discriminator_features)
        self.assertEqual(gan.output_shape, self.output_shape)

    def test_generator_forward_pass(self):
        gan = GAN(
            latent_dim=self.latent_dim,
            generator_features=self.generator_features,
            discriminator_features=self.discriminator_features,
            output_shape=self.output_shape
        )
        rng, init_rng = jax.random.split(self.rng)
        x = jax.random.normal(rng, (self.batch_size,) + self.output_shape)
        variables = gan.init(init_rng, x, init_rng)
        rng, gen_rng = jax.random.split(rng)
        generated_images, _, _ = gan.apply(variables, x, rng=gen_rng, train=True)

        self.assertEqual(generated_images.shape, (self.batch_size,) + self.output_shape)

    def test_discriminator_forward_pass(self):
        gan = GAN(
            latent_dim=self.latent_dim,
            generator_features=self.generator_features,
            discriminator_features=self.discriminator_features,
            output_shape=self.output_shape
        )
        # Create fake input data with the same shape as the output
        x = jax.random.normal(self.rng, (self.batch_size,) + self.output_shape)
        rng, init_rng = jax.random.split(self.rng)

        variables = gan.init(init_rng, x, init_rng)
        rng, apply_rng = jax.random.split(rng)
        generated, real_output, fake_output = gan.apply(variables, x, rng=apply_rng, train=True)

        self.assertEqual(generated.shape, (self.batch_size,) + self.output_shape)
        self.assertEqual(real_output.shape, (self.batch_size, 1))
        self.assertEqual(fake_output.shape, (self.batch_size, 1))

    def test_generator_loss(self):
        gan = GAN(
            latent_dim=self.latent_dim,
            generator_features=self.generator_features,
            discriminator_features=self.discriminator_features,
            output_shape=self.output_shape
        )
        fake_logits = jax.random.normal(self.rng, (self.batch_size, 1))
        loss = gan.generator_loss(fake_logits)

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())
        self.assertTrue(jnp.all(loss >= 0))  # Generator loss should be non-negative

    def test_discriminator_loss(self):
        gan = GAN(
            latent_dim=self.latent_dim,
            generator_features=self.generator_features,
            discriminator_features=self.discriminator_features,
            output_shape=self.output_shape
        )
        real_logits = jax.random.normal(self.rng, (self.batch_size, 1))
        fake_logits = jax.random.normal(self.rng, (self.batch_size, 1))
        loss = gan.discriminator_loss(real_logits, fake_logits)

        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())
        self.assertGreater(loss, 0)  # Loss should be positive

if __name__ == '__main__':
    unittest.main()
