import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

class GAN(nn.Module):
    """
    Generative Adversarial Network (GAN) module.

    This module implements the basic structure of a GAN, including
    a generator and a discriminator.
    """
    latent_dim: int
    generator_features: Tuple[int, ...]
    discriminator_features: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    activation: Callable = nn.relu

    def setup(self):
        # Generator
        self.generator = self.create_generator()

        # Discriminator
        self.discriminator = self.create_discriminator()

    def create_generator(self):
        return nn.Sequential([
            nn.Dense(features) for features in self.generator_features
        ] + [nn.Dense(jnp.prod(self.output_shape))])

    def create_discriminator(self):
        return nn.Sequential([
            nn.Dense(features) for features in self.discriminator_features
        ] + [nn.Dense(1)])

    def generate(self, z):
        x = self.generator(z)
        return x.reshape((-1,) + self.output_shape)

    def discriminate(self, x):
        x_flat = x.reshape((x.shape[0], -1))
        return self.discriminator(x_flat)

    def generator_loss(self, fake_output):
        return -jnp.mean(jnp.log(fake_output + 1e-8))

    def discriminator_loss(self, real_output, fake_output):
        real_loss = -jnp.mean(jnp.log(real_output + 1e-8))
        fake_loss = -jnp.mean(jnp.log(1 - fake_output + 1e-8))
        return real_loss + fake_loss

# Example usage:
# gan = GAN(latent_dim=100,
#           generator_features=(128, 256, 512),
#           discriminator_features=(512, 256, 128),
#           output_shape=(28, 28, 1))
