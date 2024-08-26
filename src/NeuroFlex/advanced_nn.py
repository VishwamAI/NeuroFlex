from typing import Sequence, Optional, Tuple, Any, Union, List, Callable, Dict
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
import logging
from jax import Array
from dataclasses import field
from functools import partial
import optax
from .rl_module import PrioritizedReplayBuffer, create_train_state, select_action, RLAgent



class NeuroFlexNN(nn.Module):
    """
    A flexible neural network module that can be configured for various tasks, including reinforcement learning.

    Args:
        features (Tuple[int, ...]): The number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        conv_dim (int, optional): The dimension of convolution (2 or 3). Defaults to 2.
        action_dim (Optional[int], optional): The dimension of the action space for RL. Defaults to None.
        use_cnn (bool, optional): Whether to use convolutional layers. Defaults to False.
        use_rl (bool, optional): Whether to use reinforcement learning components. Defaults to False.
        use_residual (bool, optional): Whether to use residual connections. Defaults to False.
        use_dueling (bool, optional): Whether to use dueling DQN architecture. Defaults to False.
        use_double (bool, optional): Whether to use double Q-learning. Defaults to False.
        dtype (Union[jnp.dtype, str], optional): The data type to use for computations. Defaults to jnp.float32.
        activation (Callable, optional): The activation function to use in the network. Defaults to nn.relu.
        max_retries (int, optional): Maximum number of retries for self-curing. Defaults to 3.
        rl_learning_rate (float, optional): Learning rate for RL components. Defaults to 1e-4.
        rl_gamma (float, optional): Discount factor for RL. Defaults to 0.99.
        rl_epsilon_start (float, optional): Starting epsilon for ε-greedy policy. Defaults to 1.0.
        rl_epsilon_end (float, optional): Ending epsilon for ε-greedy policy. Defaults to 0.01.
        rl_epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.
    """
    features: Tuple[int, ...]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    conv_dim: int = 2
    action_dim: Optional[int] = None
    use_cnn: bool = False
    use_rl: bool = False
    use_residual: bool = False
    use_dueling: bool = False
    use_double: bool = False
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
        self.step_count = self.variable('state', 'step_count', lambda: jnp.array(0, dtype=jnp.int32))
        self.rng = self.make_rng('params')

        if self.use_cnn:
            self._setup_cnn_layers()
            self.cnn_block = self._create_cnn_block()

        self._setup_dense_layers()

        if self.use_rl:
            if self.use_dueling:
                self.value_stream = nn.Dense(1, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())
                self.advantage_stream = nn.Dense(self.action_dim, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())
            else:
                self.rl_layer = nn.Dense(self.action_dim, dtype=self.dtype, kernel_init=nn.initializers.kaiming_normal())

            if self.use_double:
                self.target_network = self._create_target_network()

            self.lr_schedule = optax.warmup_cosine_decay_schedule(
                init_value=self.rl_learning_rate / 10,
                peak_value=self.rl_learning_rate,
                warmup_steps=1000,
                decay_steps=10000,
                end_value=self.rl_learning_rate / 100
            )
            self.rl_optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(learning_rate=self.lr_schedule)
            )
            self.replay_buffer = PrioritizedReplayBuffer(100000)  # Using PrioritizedReplayBuffer
            self.rl_epsilon = self.variable('state', 'rl_epsilon', lambda: jnp.array(self.rl_epsilon_start, dtype=jnp.float32))
        else:
            self.final_dense = nn.Dense(
                self.output_shape[-1],
                dtype=self.dtype,
                kernel_init=nn.initializers.kaiming_normal()
            )

        # Initialize params attribute
        self.params = self.param('params', self._init_params)

    def _init_params(self, key):
        """Initialize the parameters of the neural network."""
        params = {}
        dummy_input = jnp.ones((1,) + self.input_shape[1:], dtype=self.dtype)

        if self.use_cnn:
            for i, feat in enumerate(self.features[:-1]):
                key, subkey = jax.random.split(key)
                kernel_shape = (3,) * self.conv_dim + (dummy_input.shape[-1] if i == 0 else self.features[i-1], feat)
                params[f'conv_layers_{i}'] = {
                    'kernel': self.param(f'conv_kernel_{i}', nn.initializers.kaiming_normal(), kernel_shape),
                    'bias': self.param(f'conv_bias_{i}', nn.initializers.zeros, (feat,))
                }
            dummy_input = dummy_input.reshape((1, -1))  # Flatten for dense layers

        for i, feat in enumerate(self.features):
            key, subkey = jax.random.split(key)
            input_dim = dummy_input.shape[-1] if i == 0 else self.features[i-1]
            params[f'dense_layers_{i}'] = {
                'kernel': self.param(f'dense_kernel_{i}', nn.initializers.kaiming_normal(), (input_dim, feat)),
                'bias': self.param(f'dense_bias_{i}', nn.initializers.zeros, (feat,))
            }

        if not self.use_rl:
            key, subkey = jax.random.split(key)
            params['final_dense'] = {
                'kernel': self.param('final_dense_kernel', nn.initializers.kaiming_normal(), (self.features[-1], self.output_shape[-1])),
                'bias': self.param('final_dense_bias', nn.initializers.zeros, (self.output_shape[-1],))
            }

        return params

        if self.use_cnn:
            self._setup_cnn_layers()
            self.cnn_block = self._create_cnn_block()

        self._setup_dense_layers()

    def _create_cnn_block(self):
        """Create the CNN block."""
        def cnn_block(x, train: bool = True):
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = conv(x)
                x = bn(x, use_running_average=not train)
                x = self.activation(x)
                x = nn.max_pool(x, window_shape=(2,) * self.conv_dim, strides=(2,) * self.conv_dim)
            return x.reshape((x.shape[0], -1))  # Flatten
        return cnn_block

    def _fallback_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generate a fallback output in case of persistent errors."""
        logging.warning("Generating fallback output.")
        return jnp.zeros(self.output_shape, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, train: bool = True, params: Optional[Dict] = None) -> jnp.ndarray:
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input tensor.
            train (bool): Whether to run in training mode. Defaults to True.
            params (Optional[Dict]): Optional parameters to use for the forward pass.

        Returns:
            jnp.ndarray: Output tensor.
        """
        variables = {'params': params if params is not None else self.params}
        for attempt in range(self.max_retries):
            try:
                self._validate_input_shape(x)
                return self._forward(x, train, variables)
            except ValueError as ve:
                logging.warning(f"Input shape mismatch (attempt {attempt + 1}/{self.max_retries}): {str(ve)}")
                x = self._attempt_recovery(x, ve)
            except Exception as e:
                logging.warning(f"Error during forward pass (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                x = self._attempt_recovery(x, e)

            if attempt == self.max_retries - 1:
                logging.error("Max retries reached. Returning fallback output.")
                return self._fallback_output(x)

        return self._fallback_output(x)

    def _validate_shapes(self):
        """Validate the input and output shapes of the network."""
        if not isinstance(self.input_shape, tuple) or not isinstance(self.output_shape, tuple):
            raise ValueError(f"Input and output shapes must be tuples. Got input_shape: {type(self.input_shape)}, output_shape: {type(self.output_shape)}")

        if len(self.input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {self.input_shape}")
        if len(self.output_shape) < 2:
            raise ValueError(f"Output shape must have at least 2 dimensions, got {self.output_shape}")

        if self.use_cnn:
            if len(self.input_shape) != self.conv_dim + 2:
                raise ValueError(f"For CNN with conv_dim={self.conv_dim}, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
            if self.input_shape[0] != self.output_shape[0]:
                raise ValueError(f"Batch size mismatch. Input shape: {self.input_shape[0]}, Output shape: {self.output_shape[0]}")

        if self.use_rl and self.action_dim is None:
            raise ValueError("action_dim must be provided when use_rl is True")

        if self.features[-1] != self.output_shape[-1]:
            raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")

        if any(dim <= 0 for dim in self.input_shape + self.output_shape):
            raise ValueError(f"All dimensions must be positive. Input shape: {self.input_shape}, Output shape: {self.output_shape}")

    def _setup_cnn_layers(self):
        self.conv_layers = [
            nn.Conv(
                features=feat,
                kernel_size=(3,) * self.conv_dim,
                dtype=self.dtype,
                padding='SAME',
                name=f"conv_{i}",
                kernel_init=nn.initializers.kaiming_normal()
            )
            for i, feat in enumerate(self.features[:-1])
        ]
        self.bn_layers = [
            nn.BatchNorm(
                dtype=self.dtype,
                name=f"bn_{i}",
                use_running_average=True,
                momentum=0.9,
                epsilon=1e-5
            )
            for i in range(len(self.features) - 1)
        ]

    def _forward(self, x: jnp.ndarray, train: bool, variables: Dict) -> jnp.ndarray:
        """Internal forward pass implementation."""
        if self.use_cnn:
            x = self.cnn_block(x, train)

        for layer in self.dense_layers:
            x = layer(x)
            x = self.activation(x)

        if self.use_rl:
            if self.use_dueling:
                value = self.value_stream(x)
                advantage = self.advantage_stream(x)
                x = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
            else:
                x = self.rl_layer(x)
        else:
            x = self.final_dense(x)

        return x

    def _attempt_recovery(self, x: jnp.ndarray, error: Exception) -> jnp.ndarray:
        """Attempt to recover from errors during forward pass."""
        if isinstance(error, ValueError) and "shape mismatch" in str(error):
            return jnp.reshape(x, self.input_shape)
        elif isinstance(error, jax.errors.InvalidArgumentError):
            return jnp.clip(x, -1e6, 1e6)
        return x

    def _validate_input_shape(self, x: jnp.ndarray) -> None:
        """Validate the shape of the input tensor."""
        if x.shape[1:] != self.input_shape[1:]:
            raise ValueError(f"Input shape mismatch. Expected {self.input_shape}, got {x.shape}")

    def _attempt_recovery(self, x: jnp.ndarray, error: Exception) -> jnp.ndarray:
        """Attempt to recover from errors during forward pass."""
        logging.warning(f"Attempting to recover from error: {str(error)}")

        if isinstance(error, ValueError):
            if "shape mismatch" in str(error):
                logging.info("Attempting to reshape input due to shape mismatch.")
                return jnp.reshape(x, self.input_shape)
            elif "invalid value" in str(error):
                logging.info("Attempting to replace invalid values with zeros.")
                return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(error, jax.errors.JAXTypeError):
            logging.info("JAX type error encountered. Attempting to cast input to float32.")
            return x.astype(jnp.float32)

        if isinstance(error, flax.errors.ScopeCollectionNotFound):
            if "batch_stats" in str(error):
                logging.info("Attempting to reinitialize batch statistics.")
                # In a real scenario, we would reinitialize batch stats here
                # For now, we'll just return the input as is
                return x

        logging.warning("No specific recovery method found. Returning original input.")
        return x

    def _setup_dense_layers(self):
        logging.debug("Setting up dense layers")
        dense_layers = []
        for i, feat in enumerate(self.features[:-1]):
            dense_layer = nn.Dense(
                feat,
                dtype=self.dtype,
                name=f"dense_{i}",
                kernel_init=nn.initializers.he_normal()
            )
            dense_layers.append(dense_layer)
            dense_layers.append(nn.BatchNorm(dtype=self.dtype, name=f"bn_dense_{i}"))

            if self.use_residual:
                dense_layers.append(ResidualBlock(dense_layer))

            dense_layers.append(self.activation)

        dense_layers.append(nn.Dropout(0.5))
        self.dense_layers = nn.Sequential(dense_layers)
        logging.debug(f"Dense layers setup completed successfully. Total layers: {len(dense_layers)}")

    def _create_target_network(self):
        """Create a target network for double Q-learning."""
        return NeuroFlexNN(
            features=self.features,
            input_shape=self.input_shape,
            output_shape=self.output_shape,
            conv_dim=self.conv_dim,
            action_dim=self.action_dim,
            use_cnn=self.use_cnn,
            use_rl=self.use_rl,
            use_residual=self.use_residual,
            use_dueling=self.use_dueling,
            use_double=False,  # Prevent infinite recursion
            dtype=self.dtype,
            activation=self.activation
        )

    def _forward(self, x: jnp.ndarray, train: bool, variables: Dict) -> jnp.ndarray:
        """Internal forward pass implementation."""
        logging.debug(f"Input shape: {x.shape}")

        try:
            self._validate_input_shape(x)
        except ValueError as e:
            logging.warning(f"Input shape mismatch: {str(e)}. Attempting to reshape.")
            x = self._reshape_input(x)
            logging.debug(f"Reshaped input shape: {x.shape}")

        if self.use_cnn:
            x = self.cnn_block(x, train)
            logging.debug(f"After CNN block shape: {x.shape}")
            if x.ndim != 2:
                x = x.reshape((x.shape[0], -1))
                logging.debug(f"Flattened CNN output shape: {x.shape}")

        x = self.dense_layers(x)
        logging.debug(f"After DNN block shape: {x.shape}")

        if self.use_rl:
            if self.use_dueling:
                value = self.value_stream(x)
                advantages = self.advantage_stream(x)
                q_values = value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
            else:
                q_values = self.rl_layer(x)

            logging.debug(f"Q-values shape: {q_values.shape}")

            if not train:
                x = jnp.argmax(q_values, axis=-1)
            else:
                epsilon = self.rl_epsilon_end + (self.rl_epsilon_start - self.rl_epsilon_end) * \
                          jnp.exp(-self.rl_epsilon_decay * self.step_count.value)
                self.sow('intermediates', 'epsilon', epsilon)
                self.rng, subkey = jax.random.split(self.rng)
                x = jax.lax.cond(
                    jax.random.uniform(subkey) < epsilon,
                    lambda: jax.random.randint(subkey, (x.shape[0],), 0, self.action_dim),
                    lambda: jnp.argmax(q_values, axis=-1)
                )
            self.step_count.value += 1
            logging.debug(f"RL action shape: {x.shape}, Epsilon: {epsilon:.4f}")
        else:
            x = self.final_dense(x)
            logging.debug(f"Final dense output shape: {x.shape}")

        if x.shape != self.output_shape:
            logging.warning(f"Output shape mismatch. Expected {self.output_shape}, got {x.shape}")
            x = jnp.reshape(x, self.output_shape)
            logging.debug(f"Reshaped output to: {x.shape}")

        if not jnp.all(jnp.isfinite(x)):
            logging.warning("Output contains non-finite values. Replacing with zeros.")
            x = jnp.where(jnp.isfinite(x), x, 0.0)
            logging.debug("Non-finite values replaced with zeros.")

        logging.debug(f"Final output shape: {x.shape}")
        return x

class ResidualBlock(nn.Module):
    layer: nn.Module

    @nn.compact
    def __call__(self, x):
        return x + self.layer(x)

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
        logging.debug(f"Input shape: {x.shape}")

        try:
            self._validate_input_shape(x)
        except ValueError as e:
            logging.warning(f"Input shape mismatch: {str(e)}. Attempting to reshape.")
            x = self._reshape_input(x)
            logging.debug(f"Reshaped input shape: {x.shape}")

        if self.use_cnn:
            x = self.cnn_block(x, deterministic)
            logging.debug(f"After CNN block shape: {x.shape}")
            if x.ndim != 2:
                x = x.reshape((x.shape[0], -1))
                logging.debug(f"Flattened CNN output shape: {x.shape}")

        x = self.dnn_block(x, deterministic)
        logging.debug(f"After DNN block shape: {x.shape}")

        if self.use_rl:
            q_values = self.rl_layer(x)
            logging.debug(f"Q-values shape: {q_values.shape}")
            if deterministic:
                x = jnp.argmax(q_values, axis=-1)
            else:
                epsilon = self.rl_epsilon_end + (self.rl_epsilon_start - self.rl_epsilon_end) * \
                          jnp.exp(-self.rl_epsilon_decay * self.step_count)
                self.rng, subkey = jax.random.split(self.rng)
                x = jax.lax.cond(
                    jax.random.uniform(subkey) < epsilon,
                    lambda: jax.random.randint(subkey, (x.shape[0],), 0, self.action_dim),
                    lambda: jnp.argmax(q_values, axis=-1)
                )
            self.step_count += 1
            logging.debug(f"RL action shape: {x.shape}, Epsilon: {epsilon:.4f}")
        else:
            x = self.final_dense(x)
            logging.debug(f"Final dense output shape: {x.shape}")

        if x.shape != self.output_shape:
            logging.warning(f"Output shape mismatch. Expected {self.output_shape}, got {x.shape}")
            x = jnp.reshape(x, self.output_shape)
            logging.debug(f"Reshaped output to: {x.shape}")

        if not jnp.all(jnp.isfinite(x)):
            logging.warning("Output contains non-finite values. Replacing with zeros.")
            x = jnp.where(jnp.isfinite(x), x, 0.0)
            logging.debug("Non-finite values replaced with zeros.")

        logging.debug(f"Final output shape: {x.shape}")
        return x

    def _attempt_recovery(self, x: jnp.ndarray, error: Exception) -> jnp.ndarray:
        """Attempt to recover from errors during forward pass."""
        logging.warning(f"Attempting to recover from error: {str(error)}")

        if isinstance(error, ValueError):
            if "shape mismatch" in str(error):
                logging.info("Attempting to reshape input due to shape mismatch.")
                return self._reshape_input(x)
            elif "invalid value" in str(error):
                logging.info("Attempting to replace invalid values with zeros.")
                return jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        if isinstance(error, jax.errors.JAXTypeError):
            logging.info("JAX type error encountered. Attempting to cast input to float32.")
            return x.astype(jnp.float32)

        if isinstance(error, flax.errors.ScopeCollectionNotFound):
            if "batch_stats" in str(error):
                logging.info("Attempting to reinitialize batch statistics.")
                # In a real scenario, we would reinitialize batch stats here
                # For now, we'll just return the input as is
                return x

        if isinstance(error, jax.errors.InvalidArgumentError):
            logging.info("Invalid argument error. Attempting to clip input values.")
            return jnp.clip(x, -1e6, 1e6)

        if isinstance(error, OverflowError):
            logging.info("Overflow error detected. Attempting to normalize input.")
            return x / jnp.linalg.norm(x)

        logging.warning("No specific recovery method found. Returning original input.")
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
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
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

def adversarial_training(model, params, input_data, epsilon, step_size):
    """
    Generate adversarial examples using the Fast Gradient Sign Method (FGSM).

    Args:
        model (NeuroFlexNN): The model to generate adversarial examples for.
        params (dict): The model parameters.
        input_data (dict): A dictionary containing 'image' and 'label' keys.
        epsilon (float): The maximum perturbation allowed.
        step_size (float): The step size for the perturbation.

    Returns:
        dict: A dictionary containing the perturbed 'image' and original 'label'.
    """
    def loss_fn(params, image, label):
        logits = model.apply({'params': params}, image)
        return optax.softmax_cross_entropy(logits, label).mean()

    grad_fn = jax.grad(loss_fn, argnums=1)
    image, label = input_data['image'], input_data['label']

    grad = grad_fn(params, image, label)
    perturbation = jnp.sign(grad) * step_size
    perturbed_image = jnp.clip(image + perturbation, 0, 1)

    # Ensure the perturbation is within epsilon
    total_perturbation = perturbed_image - image
    total_perturbation = jnp.clip(total_perturbation, -epsilon, epsilon)
    perturbed_image = image + total_perturbation

    return {'image': perturbed_image, 'label': label}
