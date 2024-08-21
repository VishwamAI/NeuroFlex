from typing import Sequence, Optional, Tuple, Any, Union, List, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
import logging
from jax import Array
from dataclasses import field
from functools import partial
import optax
from .rl_module import ReplayBuffer, create_train_state, select_action

class NeuroFlexNN(nn.Module):
    """
    A flexible neural network module that can be configured for various tasks, including reinforcement learning.

    Args:
        features (List[int]): The number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        conv_dim (int, optional): The dimension of convolution (2 or 3). Defaults to 2.
        action_dim (Optional[int], optional): The dimension of the action space for RL. Defaults to None.
        use_cnn (bool, optional): Whether to use convolutional layers. Defaults to False.
        use_rl (bool, optional): Whether to use reinforcement learning components. Defaults to False.
        dtype (Union[jnp.dtype, str], optional): The data type to use for computations. Defaults to jnp.float32.
        activation (Callable, optional): The activation function to use in the network. Defaults to nn.relu.
        max_retries (int, optional): Maximum number of retries for self-curing. Defaults to 3.
        rl_learning_rate (float, optional): Learning rate for RL components. Defaults to 1e-4.
        rl_gamma (float, optional): Discount factor for RL. Defaults to 0.99.
        rl_epsilon_start (float, optional): Starting epsilon for ε-greedy policy. Defaults to 1.0.
        rl_epsilon_end (float, optional): Ending epsilon for ε-greedy policy. Defaults to 0.01.
        rl_epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.

    Attributes:
        features (List[int]): The number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        conv_dim (int): The dimension of convolution (2 or 3).
        action_dim (Optional[int]): The dimension of the action space for RL.
        use_cnn (bool): Whether to use convolutional layers.
        use_rl (bool): Whether to use reinforcement learning components.
        dtype (jnp.dtype): The data type to use for computations.
        activation (Callable): The activation function to use in the network.
        max_retries (int): Maximum number of retries for self-curing.
        rl_learning_rate (float): Learning rate for RL components.
        rl_gamma (float): Discount factor for RL.
        rl_epsilon_start (float): Starting epsilon for ε-greedy policy.
        rl_epsilon_end (float): Ending epsilon for ε-greedy policy.
        rl_epsilon_decay (float): Decay rate for epsilon.
    """
    features: List[int]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    conv_dim: int = 2
    action_dim: Optional[int] = None
    use_cnn: bool = False
    use_rl: bool = False
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu
    max_retries: int = 3
    rl_learning_rate: float = 1e-4
    rl_gamma: float = 0.99
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.01
    rl_epsilon_decay: float = 0.995

    def setup(self):
        """Initialize the layers of the neural network."""
        self._validate_shapes()
        self.conv_layers = []
        self.bn_layers = []
        self.dense_layers = []
        self.step_count = 0
        self.rng = jax.random.PRNGKey(0)  # Initialize with a default seed

        if self.use_cnn:
            self._setup_cnn_layers()
        self._setup_dense_layers()

        self.final_dense = nn.Dense(self.output_shape[-1], dtype=self.dtype, name="final_dense")
        if self.use_rl:
            self.rl_layer = nn.Dense(self.action_dim, dtype=self.dtype, name="rl_layer")
            self.value_layer = nn.Dense(1, dtype=self.dtype, name="value_layer")
            self.rl_optimizer = optax.adam(learning_rate=self.rl_learning_rate)
            self.replay_buffer = ReplayBuffer(100000)  # Default buffer size of 100,000
            self.rl_epsilon = self.rl_epsilon_start

        if self.use_cnn:
            self._setup_cnn_layers()
        self._setup_dense_layers()

        self.final_dense = nn.Dense(self.output_shape[-1], dtype=self.dtype, name="final_dense")
        if self.use_rl:
            self.rl_layer = nn.Dense(self.action_dim, dtype=self.dtype, name="rl_layer")
            self.value_layer = nn.Dense(1, dtype=self.dtype, name="value_layer")
            self.rl_optimizer = optax.adam(learning_rate=self.rl_learning_rate)
            self.replay_buffer = ReplayBuffer(100000)  # Default buffer size of 100,000
            self.rl_epsilon = self.rl_epsilon_start

    def _setup_cnn_layers(self):
        self.conv_layers = [nn.Conv(features=feat, kernel_size=(3,) * self.conv_dim, dtype=self.dtype, padding='SAME', name=f"conv_{i}")
                            for i, feat in enumerate(self.features[:-1])]
        self.bn_layers = [nn.BatchNorm(dtype=self.dtype, name=f"bn_{i}")
                          for i in range(len(self.features) - 1)]

    def _setup_dense_layers(self):
        self.dense_layers = [nn.Dense(feat, dtype=self.dtype, name=f"dense_{i}")
                             for i, feat in enumerate(self.features[:-1])]
        self.dense_layers.append(nn.Dropout(0.5))

    def _validate_shapes(self):
        """Validate the input and output shapes of the network."""
        if len(self.input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {self.input_shape}")
        if len(self.output_shape) < 2:
            raise ValueError(f"Output shape must have at least 2 dimensions, got {self.output_shape}")
        if self.use_cnn and len(self.input_shape) != self.conv_dim + 2:
            raise ValueError(f"For CNN, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
        if self.use_rl and self.action_dim is None:
            raise ValueError("action_dim must be provided when use_rl is True")
        if self.features[-1] != self.output_shape[-1]:
            raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")

    def __call__(self, x: jnp.ndarray, deterministic: bool = False) -> jnp.ndarray:
        """
        Forward pass of the neural network with self-curing mechanism.

        Args:
            x (jnp.ndarray): Input tensor.
            deterministic (bool): Whether to run in deterministic mode (e.g., for inference).

        Returns:
            jnp.ndarray: Output tensor.
        """
        for attempt in range(self.max_retries):
            try:
                return self._forward(x, deterministic)
            except Exception as e:
                logging.warning(f"Error during forward pass (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                x = self._attempt_recovery(x, e)
                if attempt == self.max_retries - 1:
                    logging.error("Max retries reached. Returning fallback output.")
                    return self._fallback_output(x)

        return self._fallback_output(x)

    def _forward(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Internal forward pass implementation."""
        self._validate_input_shape(x)

        if self.use_cnn:
            x = self.cnn_block(x, deterministic)
        x = self.dnn_block(x, deterministic)

        if self.use_rl:
            q_values = self.rl_layer(x)
            if deterministic:
                x = jnp.argmax(q_values, axis=-1)
            else:
                epsilon = self.rl_epsilon_end + (self.rl_epsilon_start - self.rl_epsilon_end) * \
                          jnp.exp(-self.rl_epsilon_decay * self.step_count)
                if not hasattr(self, 'rng'):
                    self.rng = jax.random.PRNGKey(0)
                self.rng, subkey = jax.random.split(self.rng)
                x = jax.lax.cond(
                    jax.random.uniform(subkey) < epsilon,
                    lambda: jax.random.randint(subkey, (x.shape[0],), 0, self.action_dim),
                    lambda: jnp.argmax(q_values, axis=-1)
                )
            self.step_count += 1
        else:
            x = self.final_dense(x)

        if x.shape != self.output_shape:
            logging.warning(f"Output shape mismatch. Expected {self.output_shape}, got {x.shape}")
            x = jnp.reshape(x, self.output_shape)

        if not jnp.all(jnp.isfinite(x)):
            logging.warning("Output contains non-finite values. Replacing with zeros.")
            x = jnp.where(jnp.isfinite(x), x, 0.0)

        return x

    def _attempt_recovery(self, x: jnp.ndarray, error: Exception) -> jnp.ndarray:
        """Attempt to recover from errors during forward pass."""
        logging.info(f"Attempting to recover from error: {str(error)}")

        if isinstance(error, ValueError) and "shape mismatch" in str(error):
            logging.info("Attempting to reshape input.")
            return jnp.reshape(x, self.input_shape)

        if isinstance(error, jax.errors.InvalidArgumentError):
            logging.info("Attempting to handle invalid argument error.")
            return jnp.clip(x, -1e6, 1e6)  # Clip to prevent overflow

        # Add more specific error handling cases as needed

        logging.info("No specific recovery method. Returning original input.")
        return x

    def _fallback_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generate a fallback output in case of persistent errors."""
        logging.warning("Generating fallback output.")
        return jnp.zeros(self.output_shape, dtype=x.dtype)

    def cnn_block(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Apply CNN layers to the input."""
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = conv(x)
            x = bn(x, use_running_average=deterministic)
            x = self.activation(x)
            x = nn.max_pool(x, window_shape=(2,) * self.conv_dim, strides=(2,) * self.conv_dim)
        return x.reshape((x.shape[0], -1))  # Flatten

    def dnn_block(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        """Apply DNN layers to the input."""
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.dense_layers[-1](x, deterministic=deterministic)  # Dropout layer
        return x

    def _validate_input_shape(self, x: jnp.ndarray) -> None:
        """Validate the shape of the input tensor."""
        if x.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"Input shape mismatch. Expected {self.input_shape}, got {x.shape}")

    def get_cnn_output_shape(self, input_shape: Sequence[int]) -> Tuple[int, ...]:
        """Calculate the output shape of the CNN layers."""
        shape = list(input_shape)
        for _ in range(len(self.features) - 1):
            shape = [max(1, s // 2) for s in shape[:-1]] + [self.features[-2]]
        return tuple(shape)

    def calculate_input_size(self) -> int:
        """Calculate the total input size for the dense layers."""
        if self.use_cnn:
            cnn_output_shape = self.get_cnn_output_shape(self.input_shape[1:])
            return int(jnp.prod(jnp.array(cnn_output_shape)))
        else:
            return int(jnp.prod(jnp.array(self.input_shape[1:])))

def create_neuroflex_nn(features: Sequence[int], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], use_cnn: bool = False, conv_dim: int = 2, use_rl: bool = False, action_dim: Optional[int] = None, dtype: Union[jnp.dtype, str] = jnp.float32) -> NeuroFlexNN:
    """
    Create a NeuroFlexNN instance with the specified features, input shape, and output shape.

    Args:
        features (Sequence[int]): A sequence of integers representing the number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        use_cnn (bool, optional): Whether to use convolutional layers. Defaults to False.
        conv_dim (int, optional): The dimension of convolution (2 or 3). Defaults to 2.
        use_rl (bool, optional): Whether to use reinforcement learning components. Defaults to False.
        action_dim (Optional[int], optional): The dimension of the action space for RL. Defaults to None.
        dtype (Union[jnp.dtype, str], optional): The data type to use for computations. Defaults to jnp.float32.

    Returns:
        NeuroFlexNN: An instance of the NeuroFlexNN class.

    Example:
        >>> model = create_neuroflex_nn([64, 32, 10], input_shape=(1, 28, 28, 1), output_shape=(1, 10), use_cnn=True)
    """
    return NeuroFlexNN(features=features, input_shape=input_shape, output_shape=output_shape,
                       use_cnn=use_cnn, conv_dim=conv_dim, use_rl=use_rl, action_dim=action_dim, dtype=dtype)

# Advanced neural network components including RL
class AdvancedNNComponents:
    def __init__(self):
        self.replay_buffer = None
        self.optimizer = None
        self.epsilon = None

    def initialize_rl_components(self, buffer_size: int, learning_rate: float, epsilon_start: float):
        self.replay_buffer = ReplayBuffer(buffer_size)
        self.optimizer = optax.adam(learning_rate)
        self.epsilon = epsilon_start

    def update_rl_model(self, state, target_state, batch):
        def loss_fn(params):
            q_values = state.apply_fn({'params': params}, batch['observations'])
            next_q_values = target_state.apply_fn({'params': target_state.params}, batch['next_observations'])
            targets = batch['rewards'] + self.gamma * jnp.max(next_q_values, axis=-1) * (1 - batch['dones'])
            loss = jnp.mean(optax.huber_loss(q_values[jnp.arange(len(batch['actions'])), batch['actions']], targets))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def select_action(self, state, observation, epsilon):
        if jax.random.uniform(jax.random.PRNGKey(0)) < epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, state.output_dim)
        else:
            q_values = state.apply_fn({'params': state.params}, observation[None, ...])
            return jnp.argmax(q_values[0])

from flax.training import train_state

def create_rl_train_state(rng, model, dummy_input, optimizer):
    params = model.init(rng, dummy_input)['params']
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
