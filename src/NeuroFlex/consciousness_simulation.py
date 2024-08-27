import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Callable, Optional
import logging
import traceback

class ConsciousnessSimulation(nn.Module):
    """
    An advanced module for simulating consciousness in the NeuroFlex framework.
    This class implements various cognitive processes and consciousness-related computations,
    including dynamic attention mechanisms, working memory, and decision-making processes.
    """

    features: List[int]
    output_dim: int
    max_attention_heads: int = 8

    def setup(self):
        self.dense_layers = [nn.Dense(feat, kernel_init=nn.initializers.xavier_uniform()) for feat in self.features]
        self.output_layer = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        self.attention_output_proj = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        self.gru_input_proj = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        self.gru_cell = nn.GRUCell(features=self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        self.decision_layer = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())
        self.metacognition_layer = nn.Dense(1, kernel_init=nn.initializers.xavier_uniform())
        self.consciousness_proj = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())
        self.thought_layer = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform())

    def _calculate_attention_heads(self, input_dim):
        return min(self.max_attention_heads, max(1, input_dim // 8))

    def _create_dynamic_attention(self, input_dim, deterministic):
        num_heads = self._calculate_attention_heads(input_dim)
        logging.debug(f"Creating dynamic attention with {num_heads} heads for input dimension {input_dim}")
        qkv_features = min(input_dim, self.output_dim)
        qkv_features = (qkv_features // num_heads) * num_heads  # Ensure qkv_features is divisible by num_heads
        out_features = self.output_dim
        logging.debug(f"qkv_features: {qkv_features}, out_features: {out_features}")
        return nn.MultiHeadDotProductAttention(
            num_heads=num_heads,
            qkv_features=qkv_features,
            out_features=out_features,
            dropout_rate=0.1,
            kernel_init=nn.initializers.xavier_uniform(),
            deterministic=deterministic,
        )

    @nn.compact
    def __call__(self, x, working_memory_state=None, generate_thought=False, deterministic=False):
        logging.debug(f"Input shape: {x.shape}")
        logging.debug(f"Generate thought: {generate_thought}")
        logging.debug(f"Deterministic: {deterministic}")

        # Ensure x is 2D
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        if x.ndim != 2:
            raise ValueError(f"Input must be 1D or 2D, got shape {x.shape}")

        batch_size, input_dim = x.shape
        logging.debug(f"Batch size: {batch_size}, Input dimension: {input_dim}")

        # Process input through dense layers
        for i, dense in enumerate(self.dense_layers):
            x = dense(x)
            x = nn.relu(x)
            logging.debug(f"After dense layer {i} shape: {x.shape}")
        cognitive_state = self.output_layer(x)
        logging.debug(f"Cognitive state shape: {cognitive_state.shape}")

        # Apply dynamic attention
        attention_output = self._apply_dynamic_attention(cognitive_state, deterministic=deterministic)
        logging.debug(f"Attention output shape: {attention_output.shape}")

        # Project the attention output
        projected_output = self.attention_output_proj(attention_output)
        logging.debug(f"Projected output shape: {projected_output.shape}")

        # Use the gru_input_proj
        gru_input = self.gru_input_proj(projected_output)
        logging.debug(f"GRU input shape: {gru_input.shape}")

        # Ensure all shapes are 2D for concatenation
        attention_output = attention_output.reshape(attention_output.shape[0], -1)
        projected_output = projected_output.reshape(projected_output.shape[0], -1)
        gru_input = gru_input.reshape(gru_input.shape[0], -1)
        logging.debug(f"Reshaped shapes - Attention: {attention_output.shape}, Projected: {projected_output.shape}, GRU: {gru_input.shape}")

        if working_memory_state is None:
            working_memory_state = jnp.zeros((batch_size, self.output_dim))
        logging.debug(f"Working memory state shape: {working_memory_state.shape}")

        new_working_memory, _ = self.gru_cell(gru_input, working_memory_state)
        logging.debug(f"New working memory shape: {new_working_memory.shape}")

        # Simulate decision-making process
        # Ensure consistent shapes for concatenation
        if cognitive_state.ndim == 3:
            cognitive_state = cognitive_state.squeeze(1)
        decision_input = jnp.concatenate([cognitive_state, gru_input, new_working_memory], axis=-1)
        logging.debug(f"Decision input shape: {decision_input.shape}")
        decision = nn.tanh(self.decision_layer(decision_input))
        decision = jnp.reshape(decision, (batch_size, 1))
        logging.debug(f"Decision shape: {decision.shape}")

        # Simulate metacognition (awareness of own cognitive processes)
        metacognition = nn.sigmoid(self.metacognition_layer(decision_input))
        metacognition = jnp.reshape(metacognition, (batch_size, 1))
        logging.debug(f"Metacognition shape: {metacognition.shape}")

        # Combine for final consciousness state
        consciousness = jnp.concatenate([
            cognitive_state,
            gru_input,
            new_working_memory,
            decision,
            metacognition
        ], axis=-1)

        logging.debug(f"Final consciousness shape: {consciousness.shape}")

        if generate_thought:
            consciousness_proj = self.consciousness_proj(consciousness)
            thought = self.thought_layer(consciousness_proj)
            thought = nn.softmax(thought, axis=-1)
            logging.debug(f"Generated thought shape: {thought.shape}")
            return thought

        return consciousness, new_working_memory, {'attention_heads': self._calculate_attention_heads(input_dim)}

    def _apply_dynamic_attention(self, x, deterministic=False):
        batch_size, input_dim = x.shape
        logging.debug(f"Applying dynamic attention - Input shape: {x.shape}, Input dimension: {input_dim}")

        try:
            # Create dynamic attention for the current input dimension
            dynamic_attention = self._create_dynamic_attention(input_dim)
            num_heads = dynamic_attention.num_heads
            qkv_features = dynamic_attention.qkv_features
            out_features = dynamic_attention.out_features
            logging.info(f"Dynamic attention parameters: num_heads={num_heads}, qkv_features={qkv_features}, out_features={out_features}")

            # Reshape input to add sequence length dimension
            x = x.reshape(batch_size, 1, input_dim)
            logging.debug(f"Reshaped input shape: {x.shape}")

            # Initialize the attention layer
            rng = self.make_rng('attention')
            variables = dynamic_attention.init(rng, x, x, x, deterministic=deterministic)

            # Apply the attention mechanism
            attention_output = dynamic_attention.apply(
                variables,
                x, x, x,  # query, key, and value are all the same for self-attention
                deterministic=deterministic,
                rngs={'dropout': self.make_rng('dropout')}
            )
            logging.debug(f"Raw attention output shape: {attention_output.shape}")

            # Reshape output back to original dimensions
            attention_output = attention_output.reshape(batch_size, -1)
            logging.debug(f"Reshaped attention output shape: {attention_output.shape}")

            # Ensure the output has the correct dimension
            if attention_output.shape[-1] != self.output_dim:
                logging.warning(f"Attention output dimension mismatch. Expected {self.output_dim}, got {attention_output.shape[-1]}. Adjusting...")
                attention_output = nn.Dense(self.output_dim, kernel_init=nn.initializers.xavier_uniform(), name="attention_dense")(attention_output)

            logging.info(f"Final attention output shape: {attention_output.shape}")
            return attention_output

        except ValueError as ve:
            logging.error(f"ValueError in dynamic attention: {str(ve)}")
            logging.debug(f"Input shape: {x.shape}, Input dimension: {input_dim}")
            raise ValueError(f"Invalid input for dynamic attention: {str(ve)}")
        except TypeError as te:
            logging.error(f"TypeError in dynamic attention: {str(te)}")
            logging.debug(f"Dynamic attention parameters: {dynamic_attention}")
            raise TypeError(f"Type mismatch in dynamic attention: {str(te)}")
        except Exception as e:
            logging.error(f"Unexpected error in dynamic attention: {str(e)}")
            logging.debug(f"Full error traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to apply dynamic attention: {str(e)}")

    def generate_thought(self, consciousness_state):
        # Simulate thought generation based on current consciousness state
        thought = self.thought_layer(consciousness_state)
        return nn.softmax(thought, axis=-1)  # Specify axis for softmax

def create_consciousness_simulation(features: List[int], output_dim: int, max_attention_heads: int = 8) -> ConsciousnessSimulation:
    """
    Create an instance of the advanced ConsciousnessSimulation module.

    Args:
        features (List[int]): List of feature dimensions for intermediate layers.
        output_dim (int): Dimension of the output layer.
        max_attention_heads (int): Maximum number of attention heads. Default is 8.

    Returns:
        ConsciousnessSimulation: An instance of the ConsciousnessSimulation class.
    """

    model = ConsciousnessSimulation(
        features=features,
        output_dim=output_dim,
        max_attention_heads=max_attention_heads
    )
    return model

# Example usage
if __name__ == "__main__":
    rng = jax.random.PRNGKey(0)
    x = jax.random.normal(rng, (1, 10))  # Example input
    model = create_consciousness_simulation(features=[64, 32], output_dim=16)
    params = model.init(rng, x)

    consciousness_state, working_memory, info = model.apply(params, x)
    thought = model.apply(params, consciousness_state, method=model.generate_thought)

    print(f"Consciousness state shape: {consciousness_state.shape}")
    print(f"Working memory shape: {working_memory.shape}")
    print(f"Number of attention heads: {info['attention_heads']}")
    print(f"Generated thought shape: {thought.shape}")
