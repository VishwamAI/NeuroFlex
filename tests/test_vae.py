import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from NeuroFlex.vae import VAE, vae_loss

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.latent_dim = 32
        self.hidden_dims = (64, 128)
        self.input_shape = (28, 28)  # Assuming MNIST-like data
        self.batch_size = 16

    def test_vae_initialization(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dims=self.hidden_dims)
        self.assertIsInstance(vae, VAE)
        self.assertEqual(vae.latent_dim, self.latent_dim)
        self.assertEqual(vae.hidden_dims, self.hidden_dims)

    def test_vae_forward_pass(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dims=self.hidden_dims)
        x = jnp.ones((self.batch_size,) + self.input_shape)
        params = vae.init(self.rng, x)['params']
        recon, mean, logvar = vae.apply({'params': params}, x, rngs={'sampling': self.rng})

        self.assertEqual(recon.shape, (self.batch_size, 784))  # Flattened output shape
        self.assertEqual(mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_vae_loss(self):
        x = jnp.ones((self.batch_size,) + self.input_shape)
        recon = jnp.ones((self.batch_size, 784))  # Flattened reconstruction
        mean = jnp.zeros((self.batch_size, self.latent_dim))
        logvar = jnp.zeros((self.batch_size, self.latent_dim))

        loss = vae_loss(recon, x.reshape(self.batch_size, -1), mean, logvar)
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())

if __name__ == '__main__':
    unittest.main()
