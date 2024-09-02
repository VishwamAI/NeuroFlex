import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

class VAE(nn.Module):
    latent_dim: int
    hidden_dims: Tuple[int, ...]
    input_shape: Tuple[int, int] = (28, 28)  # Default to MNIST-like data
    activation: Callable = nn.relu

    def setup(self):
        # Encoder
        self.encoder_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.mean_layer = nn.Dense(self.latent_dim)
        self.logvar_layer = nn.Dense(self.latent_dim)

        # Decoder
        self.decoder_layers = [nn.Dense(dim) for dim in reversed(self.hidden_dims)]
        self.output_layer = nn.Dense(self.input_shape[0] * self.input_shape[1])

    def encode(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        for layer in self.encoder_layers:
            x = self.activation(layer(x))
        mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        return mean, logvar

    def decode(self, z):
        for layer in self.decoder_layers:
            z = self.activation(layer(z))
        output = self.output_layer(z)
        return output.reshape((z.shape[0], -1))  # Ensure output is flattened

    def reparameterize(self, mean, logvar):
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(self.make_rng('sampling'), mean.shape)
        return mean + eps * std

    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))  # Flatten the input
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar

def vae_loss(recon_x, x, mean, logvar):
    x = x.reshape((x.shape[0], -1))  # Flatten the input
    bce_loss = jnp.sum(jnp.square(recon_x - x))
    kld_loss = -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
    return bce_loss + kld_loss
