import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from typing import Tuple

class CDSTDP:
    def __init__(self, learning_rate: float = 0.01, time_window: float = 20.0):
        self.learning_rate = learning_rate
        self.time_window = time_window

    @staticmethod
    @jit
    def stdp_window(delta_t: float) -> float:
        """STDP learning window function."""
        return jnp.where(delta_t >= 0,
                         jnp.exp(-delta_t / 20.0),
                         -0.5 * jnp.exp(delta_t / 20.0))

    @staticmethod
    @jit
    def consciousness_coefficient(synaptic_activity: float) -> float:
        """Calculate consciousness coefficient based on synaptic activity."""
        return jnp.tanh(synaptic_activity)

    @jit
    def update_weights(self, weights: jnp.ndarray, pre_spikes: jnp.ndarray, post_spikes: jnp.ndarray, synaptic_activity: jnp.ndarray) -> jnp.ndarray:
        """Update synaptic weights based on CD-STDP."""
        def weight_update(w, pre, post, activity):
            delta_t = post - pre
            stdp = self.stdp_window(delta_t)
            cc = self.consciousness_coefficient(activity)
            return w + self.learning_rate * stdp * cc

        return vmap(weight_update)(weights, pre_spikes, post_spikes, synaptic_activity)

def test_cdstdp():
    """Simple test function for CD-STDP model."""
    cdstdp = CDSTDP()
    
    # Generate sample data
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(key, (10,))
    pre_spikes = jax.random.uniform(key, (10,)) * 100
    post_spikes = jax.random.uniform(key, (10,)) * 100
    synaptic_activity = jax.random.uniform(key, (10,))

    # Update weights
    new_weights = cdstdp.update_weights(weights, pre_spikes, post_spikes, synaptic_activity)

    print("Initial weights:", weights)
    print("Updated weights:", new_weights)

if __name__ == "__main__":
    test_cdstdp()
