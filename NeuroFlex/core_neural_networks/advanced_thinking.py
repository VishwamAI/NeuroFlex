import jax
import jax.numpy as jnp
from jax import random, grad, jit, vmap
from flax import linen as nn
from flax.training import train_state
import optax
from NeuroFlex.utils import normalize_data, preprocess_data
from NeuroFlex.utils import calculate_descriptive_statistics

class CDSTDP:
    def __init__(self):
        self.learning_rate = 0.01
        self.time_window = 20
        self.a_plus = 0.1
        self.a_minus = 0.12
        self.tau_plus = 20.0
        self.tau_minus = 20.0

    @jit
    def update_weights(self, params, inputs, conscious_state, feedback, learning_rate):
        def update_layer(w, x, cs, fb):
            pre_synaptic = jnp.expand_dims(x, axis=1)
            post_synaptic = jnp.expand_dims(cs, axis=0)

            delta_t = jnp.arange(-self.time_window, self.time_window + 1)

            stdp = jnp.where(
                delta_t > 0,
                self.a_plus * jnp.exp(-delta_t / self.tau_plus),
                -self.a_minus * jnp.exp(delta_t / self.tau_minus)
            )

            dw = jnp.outer(pre_synaptic, post_synaptic) * stdp

            return w + learning_rate * dw * fb

        return jax.tree_map(lambda w, x: update_layer(w, x, conscious_state, feedback), params, inputs)

def create_cdstdp():
    return CDSTDP()
