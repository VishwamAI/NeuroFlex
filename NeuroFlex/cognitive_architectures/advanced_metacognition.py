import jax.numpy as jnp
import flax.linen as nn

class AdvancedMetacognition(nn.Module):
    @nn.compact
    def __call__(self, x):
        uncertainty = nn.Dense(1)(x)
        confidence = nn.Dense(1)(x)
        return jnp.concatenate([uncertainty, confidence], axis=-1)

def create_advanced_metacognition():
    return AdvancedMetacognition()
