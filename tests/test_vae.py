import unittest
import jax
import jax.numpy as jnp
import flax.linen as nn

from NeuroFlex.generative_models.vae import VAE

class TestVAE(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(0)
        self.latent_dim = 32
        self.hidden_dim = 64
        self.input_shape = (28, 28, 1)  # Assuming MNIST-like data
        self.batch_size = 16

    def test_vae_initialization(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, input_shape=self.input_shape)
        self.assertIsInstance(vae, VAE)
        self.assertEqual(vae.latent_dim, self.latent_dim)
        self.assertEqual(vae.hidden_dim, self.hidden_dim)
        self.assertEqual(vae.input_shape, self.input_shape)

    def test_vae_forward_pass(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, input_shape=self.input_shape)
        x = jnp.ones((self.batch_size,) + self.input_shape)
        rng_key = jax.random.PRNGKey(0)
        params = vae.init(rng_key, x, rng_key)['params']
        recon, mean, logvar = vae.apply({'params': params}, x, rng_key)

        self.assertEqual(recon.shape, (self.batch_size,) + self.input_shape)
        self.assertEqual(mean.shape, (self.batch_size, self.latent_dim))
        self.assertEqual(logvar.shape, (self.batch_size, self.latent_dim))

    def test_vae_loss(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, input_shape=self.input_shape)
        x = jnp.ones((self.batch_size,) + self.input_shape)
        rng_key = jax.random.PRNGKey(0)
        params = vae.init(rng_key, x, rng_key)['params']
        recon, mean, logvar = vae.apply({'params': params}, x, rng_key)

        loss = vae.loss_function(recon, x, mean, logvar)
        self.assertIsInstance(loss, jnp.ndarray)
        self.assertEqual(loss.shape, ())

    def test_kl_divergence(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, input_shape=self.input_shape)
        mean = jnp.zeros((self.batch_size, self.latent_dim))
        logvar = jnp.zeros((self.batch_size, self.latent_dim))
        kl_div = vae.kl_divergence(mean, logvar)
        self.assertEqual(kl_div.shape, (self.batch_size,))

    def test_reconstruction_loss(self):
        vae = VAE(latent_dim=self.latent_dim, hidden_dim=self.hidden_dim, input_shape=self.input_shape)
        x = jnp.ones((self.batch_size,) + self.input_shape)
        x_recon = jnp.ones((self.batch_size,) + self.input_shape)
        recon_loss = vae.reconstruction_loss(x_recon, x)
        self.assertEqual(recon_loss.shape, (self.batch_size,))

if __name__ == '__main__':
    unittest.main()
