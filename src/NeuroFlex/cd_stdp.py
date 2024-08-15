import jax
import jax.numpy as jnp
from jax import jit, vmap
from typing import Tuple

class CDSTDP:
    """
    Consciousness-Driven Spike-Timing-Dependent Plasticity (CD-STDP) class.

    This class implements a modified STDP algorithm that incorporates
    consciousness-related factors into synaptic weight updates.

    Attributes:
        learning_rate (float): The learning rate for weight updates.
        time_window (float): The time window for STDP in milliseconds.
    """

    def __init__(self, learning_rate: float = 0.01, time_window: float = 20.0):
        """
        Initialize the CD-STDP model.

        Args:
            learning_rate (float): The learning rate for weight updates. Default is 0.01.
            time_window (float): The time window for STDP in milliseconds. Default is 20.0 ms.
        """
        self.learning_rate = learning_rate
        self.time_window = time_window

    @staticmethod
    @jit
    def stdp_window(delta_t: float) -> float:
        """
        Compute the STDP learning window function.

        Args:
            delta_t (float): The time difference between post-synaptic and pre-synaptic spikes.

        Returns:
            float: The STDP weight change factor.
        """
        return jnp.where(delta_t >= 0,
                         jnp.exp(-delta_t / 20.0),
                         -0.5 * jnp.exp(delta_t / 20.0))

    @jit
    def update_weights(self, weights: jnp.ndarray, pre_spikes: jnp.ndarray,
                       post_spikes: jnp.ndarray, consciousness_factor: jnp.ndarray) -> jnp.ndarray:
        """
        Update synaptic weights based on CD-STDP.

        Args:
            weights (jnp.ndarray): Current synaptic weights.
            pre_spikes (jnp.ndarray): Pre-synaptic spike times.
            post_spikes (jnp.ndarray): Post-synaptic spike times.
            consciousness_factor (jnp.ndarray): Consciousness-related modulation factor.

        Returns:
            jnp.ndarray: Updated synaptic weights.
        """
        def weight_update(w, pre, post, cf):
            delta_t = post - pre
            stdp = self.stdp_window(delta_t)
            return w + self.learning_rate * stdp * cf

        return vmap(weight_update)(weights, pre_spikes, post_spikes, consciousness_factor)

def test_cdstdp():
    """Test function for CD-STDP model."""
    cdstdp = CDSTDP()

    # Generate sample data
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(key, (10,))
    pre_spikes = jax.random.uniform(key, (10,)) * 100
    post_spikes = jax.random.uniform(key, (10,)) * 100
    consciousness_factor = jax.random.uniform(key, (10,))

    # Update weights
    new_weights = cdstdp.update_weights(weights, pre_spikes, post_spikes, consciousness_factor)

    print("Initial weights:", weights)
    print("Updated weights:", new_weights)

if __name__ == "__main__":
    test_cdstdp()
