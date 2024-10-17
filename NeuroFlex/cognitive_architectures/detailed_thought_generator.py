import jax.numpy as jnp
import flax.linen as nn

class DetailedThoughtGenerator(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=64)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=self.output_dim)(x)
        return x

def create_detailed_thought_generator(output_dim):
    return DetailedThoughtGenerator(output_dim=output_dim)
