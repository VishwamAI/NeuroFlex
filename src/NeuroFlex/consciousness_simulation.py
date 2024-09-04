import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
import logging

from flax import linen as nn
from typing import List

import logging

class ConsciousnessSimulation(nn.Module):
    """
    An advanced module for simulating consciousness in the NeuroFlex framework.
    This class implements various cognitive processes and consciousness-related computations,
    including attention mechanisms, working memory, and decision-making processes.
    """

    features: List[int]
    output_dim: int
    working_memory_size: int = 64
    attention_heads: int = 4
    qkv_features: int = 64  # Dimension of query, key, and value for attention mechanism
    dropout_rate: float = 0.1  # Dropout rate for attention mechanism

    @nn.compact
    def __call__(self, x, deterministic: bool = True, rng: jax.random.PRNGKey = None):
        logging.debug(f"ConsciousnessSimulation called with input shape: {x.shape}")

        # Ensure input shape is (batch_size, input_dim)
        if len(x.shape) == 1:
            x = jnp.expand_dims(x, axis=0)

        for i, feat in enumerate(self.features):
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
            logging.debug(f"After dense layer {i}, shape: {x.shape}")

        cognitive_state = nn.Dense(self.output_dim)(x)
        logging.debug(f"Cognitive state shape: {cognitive_state.shape}")

        # Reshape cognitive_state to (batch_size, 1, output_dim) for attention
        cognitive_state_reshaped = jnp.expand_dims(cognitive_state, axis=1)

        attention_output = nn.MultiHeadDotProductAttention(
            num_heads=self.attention_heads,
            qkv_features=self.qkv_features,
            out_features=self.working_memory_size,
            dropout_rate=self.dropout_rate,
            kernel_init=nn.initializers.xavier_uniform()
        )(cognitive_state_reshaped, cognitive_state_reshaped, cognitive_state_reshaped, deterministic=deterministic)

        # Squeeze the attention output back to (batch_size, working_memory_size)
        attention_output = jnp.squeeze(attention_output, axis=1)
        logging.debug(f"Attention output shape: {attention_output.shape}")

        # Use Flax's variable method for managing working memory
        working_memory_state = self.variable('working_memory', 'state',
                                             lambda: jnp.zeros((x.shape[0], self.working_memory_size)))

        gru_cell = nn.GRUCell(self.working_memory_size)
        new_working_memory, _ = gru_cell(attention_output, working_memory_state.value)

        # Update working memory state
        working_memory_state.value = new_working_memory

        # Add a small perturbation to ensure working memory changes between forward passes
        if rng is not None:
            perturbation = jax.random.normal(rng, new_working_memory.shape) * 1e-6
            new_working_memory = new_working_memory + perturbation
        else:
            logging.warning("No RNG key provided for perturbation. Working memory may not change between forward passes.")

        logging.debug(f"New working memory shape: {new_working_memory.shape}")

        decision_input = jnp.concatenate([cognitive_state, attention_output, new_working_memory], axis=-1)
        decision = nn.tanh(nn.Dense(1)(decision_input))
        metacognition = nn.sigmoid(nn.Dense(1)(decision_input))

        consciousness = jnp.concatenate([
            cognitive_state,
            attention_output,
            new_working_memory,
            decision,
            metacognition
        ], axis=-1)

        logging.debug(f"Final consciousness state shape: {consciousness.shape}")

        # Ensure the consciousness state has the expected shape (batch_size, 146)
        expected_shape = (x.shape[0], self.output_dim + self.working_memory_size + self.working_memory_size + 2)
        assert consciousness.shape == expected_shape, f"Expected shape {expected_shape}, got {consciousness.shape}"

        return consciousness, new_working_memory

    def simulate_consciousness(self, x, rngs: Dict[str, Any], deterministic: bool = True):
        try:
            logging.info(f"Simulating consciousness with input shape: {x.shape}")
            logging.debug(f"PRNG keys: {rngs}")
            logging.debug(f"Deterministic mode: {deterministic}")

            if not isinstance(rngs, dict):
                raise ValueError("rngs must be a dictionary")

            consciousness, new_working_memory = self.__call__(x, deterministic=deterministic, rng=rngs.get('perturbation'))

            logging.info(f"Consciousness shape: {consciousness.shape}")
            logging.info(f"New working memory shape: {new_working_memory.shape}")

            # Apply perturbation to working memory using the provided PRNG key
            if 'perturbation' in rngs:
                perturbation = jax.random.normal(rngs['perturbation'], new_working_memory.shape) * 1e-6
                new_working_memory = new_working_memory + perturbation
                logging.info("Perturbation applied to working memory")
            else:
                logging.warning("No 'perturbation' key in rngs. Skipping perturbation.")

            working_memory = {'state': new_working_memory}
            return consciousness, new_working_memory, working_memory
        except ValueError as ve:
            logging.error(f"ValueError in simulate_consciousness: {str(ve)}")
            raise
        except TypeError as te:
            logging.error(f"TypeError in simulate_consciousness: {str(te)}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error in simulate_consciousness: {str(e)}")
            raise

    @nn.compact
    def generate_thought(self, consciousness_state):
        # Simulate thought generation based on current consciousness state
        logging.debug(f"Generate thought input shape: {consciousness_state.shape}")
        # Ensure the input shape is correct (batch_size, 146)
        assert consciousness_state.shape[1] == self.output_dim * 2 + self.working_memory_size + 2, \
            f"Expected input shape (batch_size, {self.output_dim * 2 + self.working_memory_size + 2}), got {consciousness_state.shape}"

        # Use two Dense layers to transform the consciousness state to the output dimension
        hidden = nn.Dense(64, kernel_init=nn.initializers.xavier_uniform())(consciousness_state)
        hidden = nn.relu(hidden)
        thought = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())(hidden)
        logging.debug(f"Generated thought shape before softmax: {thought.shape}")
        thought = nn.softmax(thought, axis=-1)
        logging.debug(f"Final generated thought shape: {thought.shape}")

        # Ensure the output shape is correct (batch_size, output_dim)
        assert thought.shape[1] == self.output_dim, f"Expected output shape (batch_size, {self.output_dim}), got {thought.shape}"
        return thought

def create_consciousness_simulation(features: List[int], output_dim: int, working_memory_size: int = 64, attention_heads: int = 4, qkv_features: int = 64, dropout_rate: float = 0.1) -> ConsciousnessSimulation:
    """
    Create an instance of the advanced ConsciousnessSimulation module.

    Args:
        features (List[int]): List of feature dimensions for intermediate layers.
        output_dim (int): Dimension of the output layer.
        working_memory_size (int): Size of the working memory. Default is 64.
        attention_heads (int): Number of attention heads. Default is 4.
        qkv_features (int): Dimension of query, key, and value for attention mechanism. Default is 64.
        dropout_rate (float): Dropout rate for attention mechanism. Default is 0.1.

    Returns:
        ConsciousnessSimulation: An instance of the ConsciousnessSimulation class.
    """
    return ConsciousnessSimulation(
        features=features,
        output_dim=output_dim,
        working_memory_size=working_memory_size,
        attention_heads=attention_heads,
        qkv_features=qkv_features,
        dropout_rate=dropout_rate
    )

# Example usage
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))  # Example input
    model = create_consciousness_simulation(features=[64, 32], output_dim=16)
    params = model.init(rng, x)

    # Create separate RNG keys for different operations
    rng_keys = {
        'dropout': jax.random.PRNGKey(1),
        'perturbation': jax.random.PRNGKey(2)
    }

    # Simulate consciousness
    consciousness_state, working_memory, _ = model.apply(
        {'params': params}, x,
        rngs=rng_keys,
        method=model.simulate_consciousness,
        mutable=['working_memory']
    )

    # Generate thought
    thought_rng = jax.random.PRNGKey(3)
    thought = model.apply(
        {'params': params}, consciousness_state,
        rngs={'dropout': thought_rng},
        method=model.generate_thought
    )

    print(f"Consciousness state shape: {consciousness_state.shape}")
    print(f"Working memory shape: {working_memory.shape}")
    print(f"Generated thought shape: {thought.shape}")
