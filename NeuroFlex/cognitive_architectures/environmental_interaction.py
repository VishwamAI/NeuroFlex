import jax.numpy as jnp
import flax.linen as nn

class EnvironmentalInteraction(nn.Module):
    @nn.compact
    def __call__(self, thought, external_stimuli):
        combined = jnp.concatenate([thought, external_stimuli], axis=-1)
        x = nn.Dense(features=64)(combined)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=thought.shape[-1])(x)
        return x

def create_environmental_interaction():
    return EnvironmentalInteraction()
