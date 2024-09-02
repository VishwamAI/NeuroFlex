import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Tuple, Callable

class Generator(nn.Module):
    features: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, z):
        x = z
        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name=f'Dense_{i}'
            )(x)
            x = self.activation(x)
        x = nn.Dense(
            int(jnp.prod(jnp.array(self.output_shape))),
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name='Output'
        )(x)
        return x.reshape((-1,) + self.output_shape)

class Discriminator(nn.Module):
    features: Tuple[int, ...]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        x = x.reshape((x.shape[0], -1))
        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                kernel_init=nn.initializers.xavier_uniform(),
                bias_init=nn.initializers.zeros,
                name=f'Dense_{i}'
            )(x)
            x = self.activation(x)
        return nn.Dense(
            1,
            kernel_init=nn.initializers.xavier_uniform(),
            bias_init=nn.initializers.zeros,
            name='Output'
        )(x)

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

    @nn.compact
    def __call__(self, x, rng, train=True):
        generator = Generator(
            features=self.generator_features,
            output_shape=self.output_shape,
            activation=self.activation
        )
        discriminator = Discriminator(
            features=self.discriminator_features,
            activation=self.activation
        )

        batch_size = x.shape[0]
        z = jax.random.normal(rng, (batch_size, self.latent_dim))

        generated = generator(z)
        real_output = discriminator(x)
        fake_output = discriminator(generated)

        return generated, real_output, fake_output

    def generate(self, variables, batch_size, rng):
        return self.apply(variables, jnp.zeros((batch_size,) + self.output_shape), rngs={'noise': rng}, method=self.__call__, mutable=False)

    def discriminate(self, variables, x):
        return self.apply(variables, x, method=self.__call__, mutable=False, mode='discriminate')

    def generator_loss(self, fake_logits):
        return jnp.mean(nn.softplus(-fake_logits))

    def discriminator_loss(self, real_logits, fake_logits):
        real_loss = jnp.mean(nn.softplus(-real_logits))
        fake_loss = jnp.mean(nn.softplus(fake_logits))
        return real_loss + fake_loss

# Example usage:
# gan = GAN(latent_dim=100,
#           generator_features=(128, 256, 512),
#           discriminator_features=(512, 256, 128),
#           output_shape=(28, 28, 1))
