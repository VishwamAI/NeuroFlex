import jax
import jax.numpy as jnp
import flax.linen as nn

class VAE(nn.Module):
    latent_dim: int
    hidden_dim: int
    input_shape: tuple

    def setup(self):
        self.encoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.latent_dim * 2)
        ])

        self.decoder = nn.Sequential([
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(self.hidden_dim),
            nn.relu,
            nn.Dense(jnp.prod(self.input_shape)),
            lambda x: x.reshape((-1,) + self.input_shape)
        ])

    def __call__(self, x, rng):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar, rng)
        return self.decode(z), mean, logvar

    def encode(self, x):
        h = self.encoder(x.reshape((x.shape[0], -1)))
        return jnp.split(h, 2, axis=-1)

    def decode(self, z):
        return self.decoder(z)

    def reparameterize(self, mean, logvar, rng):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(rng, mean.shape)
        return mean + eps * std

    def kl_divergence(self, mean, logvar):
        return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1)

    def reconstruction_loss(self, x_recon, x):
        return jnp.sum(jnp.square(x_recon - x), axis=(1, 2, 3))

    def loss_function(self, x_recon, x, mean, logvar):
        kl_div = self.kl_divergence(mean, logvar)
        recon_loss = self.reconstruction_loss(x_recon, x)
        return jnp.mean(recon_loss + kl_div)
