import jax
import jax.numpy as jnp
import flax.linen as nn
import logging
import sys

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class AdvancedWorkingMemory(nn.Module):
    features: int

    def initialize_state(self, batch_size):
        logging.debug(f"Initializing state with batch_size: {batch_size}, features: {self.features}")
        return (jnp.zeros((batch_size, self.features)), jnp.zeros((batch_size, self.features)))

    @nn.compact
    def __call__(self, x, carry=None):
        logging.debug(f"AdvancedWorkingMemory __call__ input shape: {x.shape}")
        batch_size, input_size = x.shape
        if input_size != self.features:
            raise ValueError(f"Input shape mismatch. Expected features: {self.features}, got: {input_size}")

        lstm = nn.LSTMCell()

        if carry is None:
            carry = self.initialize_state(batch_size)

        logging.debug(f"Carry shape before LSTM: {carry[0].shape}, {carry[1].shape}")

        new_carry, output = lstm(carry, x)
        new_c, new_h = new_carry

        logging.debug(f"LSTM output - new_c shape: {new_c.shape}, new_h shape: {new_h.shape}")
        logging.debug(f"LSTM output - output shape: {output.shape}")

        if not (isinstance(new_carry, tuple) and len(new_carry) == 2):
            raise ValueError("new_carry must be a tuple of (new_c, new_h)")

        return new_carry, output

def create_advanced_working_memory(features):
    logging.info(f"Creating AdvancedWorkingMemory with features: {features}")
    instance = AdvancedWorkingMemory(features=features)
    logging.info(f"AdvancedWorkingMemory instance created successfully")
    return instance
