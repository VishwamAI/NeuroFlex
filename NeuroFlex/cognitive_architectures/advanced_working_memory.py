import jax.numpy as jnp
import flax.linen as nn
from flax.linen import LSTMCell
import logging
import sys
import inspect

logging.basicConfig(level=logging.DEBUG, stream=sys.stdout, format='%(asctime)s - %(levelname)s - %(message)s')

class CustomLSTMCell(nn.Module):
    gate_fn: callable = nn.sigmoid
    activation_fn: callable = nn.tanh
    kernel_init: callable = nn.initializers.lecun_normal()
    recurrent_kernel_init: callable = nn.initializers.orthogonal()
    bias_init: callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, carry, inputs):
        logging.debug(f"CustomLSTMCell input - inputs shape: {inputs.shape}, carry type: {type(carry)}")
        if not (isinstance(carry, tuple) and len(carry) == 2):
            logging.error(f"Invalid carry format in CustomLSTMCell: {carry}")
            raise ValueError("Carry must be a tuple of (c, h)")

        c, h = carry
        logging.debug(f"CustomLSTMCell carry - c shape: {c.shape}, h shape: {h.shape}")

        lstm = LSTMCell(
            gate_fn=self.gate_fn,
            activation_fn=self.activation_fn,
            kernel_init=self.kernel_init,
            recurrent_kernel_init=self.recurrent_kernel_init,
            bias_init=self.bias_init
        )
        new_carry, y = lstm(carry, inputs)
        new_c, new_h = new_carry
        logging.debug(f"CustomLSTMCell output - new_carry type: {type(new_carry)}, new_c shape: {new_c.shape}, new_h shape: {new_h.shape}")
        logging.debug(f"CustomLSTMCell output - y shape: {y.shape}")
        return new_carry, y

class AdvancedWorkingMemory(nn.Module):
    memory_size: int

    def initialize_state(self, batch_size):
        logging.debug(f"Initializing state with batch_size: {batch_size}")
        initial_state = (jnp.zeros((batch_size, self.memory_size)), jnp.zeros((batch_size, self.memory_size)))
        logging.debug(f"Initial state shapes: {initial_state[0].shape}, {initial_state[1].shape}")
        return initial_state

    @nn.compact
    def __call__(self, x, state=None):
        try:
            logging.debug(f"Entering AdvancedWorkingMemory.__call__ - caller: {inspect.stack()[1].function}")
            logging.debug(f"AdvancedWorkingMemory input shape: {x.shape}, type: {type(x)}")
            logging.debug(f"Input stats: min={jnp.min(x)}, max={jnp.max(x)}, mean={jnp.mean(x)}")

            batch_size = x.shape[0]
            if state is None:
                state = self.initialize_state(batch_size)
            logging.debug(f"State shape: {state[0].shape}, {state[1].shape}, type: {type(state)}")

            if not (isinstance(state, tuple) and len(state) == 2):
                logging.error(f"Invalid state format: {state}")
                raise ValueError("State must be a tuple of (c, h)")

            c, h = state
            logging.debug(f"State components - c shape: {c.shape}, h shape: {h.shape}")
            logging.debug(f"c stats: min={jnp.min(c)}, max={jnp.max(c)}, mean={jnp.mean(c)}")
            logging.debug(f"h stats: min={jnp.min(h)}, max={jnp.max(h)}, mean={jnp.mean(h)}")

            lstm = CustomLSTMCell()
            try:
                new_state, y = lstm((c, h), x)
                logging.debug(f"LSTM output - new_state type: {type(new_state)}, y shape: {y.shape}")
                if isinstance(new_state, tuple) and len(new_state) == 2:
                    new_c, new_h = new_state
                    logging.debug(f"new_c shape: {new_c.shape}, new_h shape: {new_h.shape}")
                    logging.debug(f"new_c stats: min={jnp.min(new_c)}, max={jnp.max(new_c)}, mean={jnp.mean(new_c)}")
                    logging.debug(f"new_h stats: min={jnp.min(new_h)}, max={jnp.max(new_h)}, mean={jnp.mean(new_h)}")
                else:
                    logging.error(f"Invalid new_state format: {new_state}")
                    raise ValueError("new_state must be a tuple of (new_c, new_h)")
                logging.debug(f"y stats: min={jnp.min(y)}, max={jnp.max(y)}, mean={jnp.mean(y)}")
            except Exception as e:
                logging.error(f"Error in LSTM cell: {str(e)}")
                raise

            logging.debug(f"Returning new_state type: {type(new_state)}, y type: {type(y)}")
            logging.debug(f"new_state shapes: {new_state[0].shape}, {new_state[1].shape}")
            logging.debug(f"y shape: {y.shape}")
            return new_state, y
        except Exception as e:
            logging.error(f"Unexpected error in AdvancedWorkingMemory.__call__: {str(e)}")
            raise

def create_advanced_working_memory(memory_size):
    return AdvancedWorkingMemory(memory_size=memory_size)
