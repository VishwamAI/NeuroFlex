from typing import Sequence, Optional, Tuple, Any, Union, List, Callable
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import logging
from jax import Array
from dataclasses import field
from functools import partial
import optax
import jax.errors
import torch
from .rl_module import PrioritizedReplayBuffer, create_train_state, select_action
from .pytorch_integration import PyTorchModel, train_pytorch_model

class NeuroFlexNN(nn.Module):
    """
    A flexible neural network module that can be configured for various tasks, including reinforcement learning.
    Supports both JAX and PyTorch backends, and hierarchical reinforcement learning.

    Args:
        features (List[int]): The number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        conv_dim (int, optional): The dimension of convolution (2 or 3). Defaults to 2.
        action_dim (Optional[int], optional): The dimension of the action space for RL. Defaults to None.
        use_cnn (bool, optional): Whether to use convolutional layers. Defaults to False.
        use_rl (bool, optional): Whether to use reinforcement learning components. Defaults to False.
        dtype (Any, optional): The data type to use for computations. Defaults to jnp.float32.
        activation (Callable, optional): The activation function to use in the network. Defaults to nn.relu.
        max_retries (int, optional): Maximum number of retries for self-curing. Defaults to 3.
        rl_learning_rate (float, optional): Learning rate for RL components. Defaults to 1e-4.
        rl_gamma (float, optional): Discount factor for RL. Defaults to 0.99.
        rl_epsilon_start (float, optional): Starting epsilon for ε-greedy policy. Defaults to 1.0.
        rl_epsilon_end (float, optional): Ending epsilon for ε-greedy policy. Defaults to 0.01.
        rl_epsilon_decay (float, optional): Decay rate for epsilon. Defaults to 0.995.
        backend (str, optional): The backend to use ('jax' or 'torch'). Defaults to 'jax'.
        num_levels (int, optional): Number of levels for hierarchical RL. Defaults to 1.
        sub_action_dims (List[int], optional): Action dimensions for each level in hierarchical RL. Defaults to None.

    Attributes:
        features (List[int]): The number of units in each layer.
        input_shape (Tuple[int, ...]): The shape of the input tensor.
        output_shape (Tuple[int, ...]): The shape of the output tensor.
        conv_dim (int): The dimension of convolution (2 or 3).
        action_dim (Optional[int]): The dimension of the action space for RL.
        use_cnn (bool): Whether to use convolutional layers.
        use_rl (bool): Whether to use reinforcement learning components.
        dtype (Any): The data type to use for computations.
        activation (Callable): The activation function to use in the network.
        max_retries (int): Maximum number of retries for self-curing.
        rl_learning_rate (float): Learning rate for RL components.
        rl_gamma (float): Discount factor for RL.
        rl_epsilon_start (float): Starting epsilon for ε-greedy policy.
        rl_epsilon_end (float): Ending epsilon for ε-greedy policy.
        rl_epsilon_decay (float): Decay rate for epsilon.
        backend (str): The backend being used ('jax' or 'torch').
        num_levels (int): Number of levels for hierarchical RL.
        sub_action_dims (List[int]): Action dimensions for each level in hierarchical RL.
    """
    features: List[int]
    input_shape: Tuple[int, ...]
    output_shape: Tuple[int, ...]
    conv_dim: int = 2
    action_dim: Optional[int] = None
    use_cnn: bool = False
    use_rl: bool = False
    dtype: Any = jnp.float32
    activation: Callable = nn.relu
    max_retries: int = 3
    rl_learning_rate: float = 1e-4
    rl_gamma: float = 0.99
    rl_epsilon_start: float = 1.0
    rl_epsilon_end: float = 0.01
    rl_epsilon_decay: float = 0.995
    backend: str = 'jax'
    num_levels: int = 1
    sub_action_dims: Optional[List[int]] = None

    def setup(self):
        """Initialize the layers of the neural network."""
        self._validate_shapes()
        self.conv_layers = []
        self.bn_layers = []

        if self.backend == 'jax':
            self.dense_layers = nn.Sequential([
                nn.Dense(feat, dtype=self.dtype, name=f"dense_{i}")
                for i, feat in enumerate(self.features[:-1])
            ])
        elif self.backend == 'torch':
            self.dense_layers = torch.nn.Sequential(*[
                torch.nn.Linear(self.features[i], feat)
                for i, feat in enumerate(self.features[1:])
            ])
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        self.step_count = 0
        self.rng = jax.random.PRNGKey(0) if self.backend == 'jax' else torch.manual_seed(0)

        if self.use_cnn:
            self._setup_cnn_layers()

        if self.backend == 'jax':
            self.final_dense = nn.Dense(self.output_shape[-1], dtype=self.dtype, name="final_dense")
        else:
            self.final_dense = torch.nn.Linear(self.features[-2], self.output_shape[-1])

        if self.use_rl:
            self._setup_rl_components()

    def _setup_cnn_layers(self):
        if self.backend == 'jax':
            self.conv_layers = [nn.Conv(features=feat, kernel_size=(3,) * self.conv_dim, dtype=self.dtype, padding='SAME', name=f"conv_{i}")
                                for i, feat in enumerate(self.features[:-1])]
            self.bn_layers = [nn.BatchNorm(dtype=self.dtype, name=f"bn_{i}")
                              for i in range(len(self.features) - 1)]
        else:
            self.conv_layers = torch.nn.ModuleList([
                torch.nn.Conv2d(self.input_shape[1], feat, kernel_size=3, padding=1)
                for feat in self.features[:-1]
            ])
            self.bn_layers = torch.nn.ModuleList([
                torch.nn.BatchNorm2d(feat) for feat in self.features[:-1]
            ])

    def _setup_rl_components(self):
        if self.backend == 'jax':
            if self.num_levels > 1:
                self.rl_layers = [nn.Dense(dim, dtype=self.dtype, name=f"rl_layer_{i}") for i, dim in enumerate(self.sub_action_dims)]
                self.value_layers = [nn.Dense(1, dtype=self.dtype, name=f"value_layer_{i}") for i in range(self.num_levels)]
            else:
                self.rl_layer = nn.Dense(self.action_dim, dtype=self.dtype, name="rl_layer")
                self.value_layer = nn.Dense(1, dtype=self.dtype, name="value_layer")
            self.rl_optimizer = optax.adam(learning_rate=self.rl_learning_rate)
        else:
            if self.num_levels > 1:
                self.rl_layers = torch.nn.ModuleList([torch.nn.Linear(self.features[-1], dim) for dim in self.sub_action_dims])
                self.value_layers = torch.nn.ModuleList([torch.nn.Linear(self.features[-1], 1) for _ in range(self.num_levels)])
            else:
                self.rl_layer = torch.nn.Linear(self.features[-1], self.action_dim)
                self.value_layer = torch.nn.Linear(self.features[-1], 1)
            self.rl_optimizer = torch.optim.Adam(self.parameters(), lr=self.rl_learning_rate)

        self.replay_buffer = PrioritizedReplayBuffer(100000)  # Default buffer size of 100,000
        self.rl_epsilon = self.rl_epsilon_start

    def _validate_shapes(self):
        """Validate the input and output shapes of the network."""
        if len(self.input_shape) < 2:
            raise ValueError(f"Input shape must have at least 2 dimensions, got {self.input_shape}")
        if len(self.output_shape) < 2:
            raise ValueError(f"Output shape must have at least 2 dimensions, got {self.output_shape}")
        if self.use_cnn and len(self.input_shape) != self.conv_dim + 2:
            raise ValueError(f"For CNN, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
        if self.use_rl:
            if self.num_levels > 1 and (self.sub_action_dims is None or len(self.sub_action_dims) != self.num_levels):
                raise ValueError("sub_action_dims must be provided and match num_levels when using hierarchical RL")
            elif self.num_levels == 1 and self.action_dim is None:
                raise ValueError("action_dim must be provided when use_rl is True and num_levels is 1")
        if self.features[-1] != self.output_shape[-1]:
            raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")

    def __call__(self, x: Union[jnp.ndarray, torch.Tensor], deterministic: bool = False) -> Union[jnp.ndarray, torch.Tensor]:
        """
        Forward pass of the neural network with self-curing mechanism.

        Args:
            x (Union[jnp.ndarray, torch.Tensor]): Input tensor.
            deterministic (bool): Whether to run in deterministic mode (e.g., for inference).

        Returns:
            Union[jnp.ndarray, torch.Tensor]: Output tensor.
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

    def _forward(self, x: Union[jnp.ndarray, torch.Tensor], deterministic: bool) -> Union[jnp.ndarray, torch.Tensor]:
        """Internal forward pass implementation."""
        self._validate_input_shape(x)

        if self.use_cnn:
            x = self.cnn_block(x, deterministic)
        x = self.dnn_block(x, deterministic)

        if self.use_rl:
            if self.num_levels > 1:
                q_values = [layer(x) for layer in self.rl_layers]
                values = [layer(x) for layer in self.value_layers]
                if deterministic:
                    actions = [jnp.argmax(q, axis=-1) if self.backend == 'jax' else torch.argmax(q, dim=-1) for q in q_values]
                else:
                    actions = []
                    for level, q in enumerate(q_values):
                        epsilon = self.rl_epsilon_end + (self.rl_epsilon_start - self.rl_epsilon_end) * \
                                  (jnp.exp(-self.rl_epsilon_decay * self.step_count) if self.backend == 'jax' else
                                   torch.exp(torch.tensor(-self.rl_epsilon_decay * self.step_count)))
                        if self.backend == 'jax':
                            self.rng, subkey = jax.random.split(self.rng)
                            action = jax.lax.cond(
                                jax.random.uniform(subkey) < epsilon,
                                lambda: jax.random.randint(subkey, (x.shape[0],), 0, self.sub_action_dims[level]),
                                lambda: jnp.argmax(q, axis=-1)
                            )
                        else:
                            action = torch.where(
                                torch.rand(x.shape[0]) < epsilon,
                                torch.randint(0, self.sub_action_dims[level], (x.shape[0],)),
                                torch.argmax(q, dim=-1)
                            )
                        actions.append(action)
                x = (actions, values)
            else:
                q_values = self.rl_layer(x)
                if deterministic:
                    x = jnp.argmax(q_values, axis=-1) if self.backend == 'jax' else torch.argmax(q_values, dim=-1)
                else:
                    epsilon = self.rl_epsilon_end + (self.rl_epsilon_start - self.rl_epsilon_end) * \
                              (jnp.exp(-self.rl_epsilon_decay * self.step_count) if self.backend == 'jax' else
                               torch.exp(torch.tensor(-self.rl_epsilon_decay * self.step_count)))
                    if self.backend == 'jax':
                        self.rng, subkey = jax.random.split(self.rng)
                        x = jax.lax.cond(
                            jax.random.uniform(subkey) < epsilon,
                            lambda: jax.random.randint(subkey, (x.shape[0],), 0, self.action_dim),
                            lambda: jnp.argmax(q_values, axis=-1)
                        )
                    else:
                        x = torch.where(
                            torch.rand(x.shape[0]) < epsilon,
                            torch.randint(0, self.action_dim, (x.shape[0],)),
                            torch.argmax(q_values, dim=-1)
                        )
            self.step_count += 1
        else:
            x = self.final_dense(x)

        if isinstance(x, tuple):
            return x
        if x.shape != self.output_shape:
            logging.warning(f"Output shape mismatch. Expected {self.output_shape}, got {x.shape}")
            x = x.reshape(self.output_shape) if self.backend == 'jax' else x.view(self.output_shape)

        if self.backend == 'jax':
            if not jnp.all(jnp.isfinite(x)):
                logging.warning("Output contains non-finite values. Replacing with zeros.")
                x = jnp.where(jnp.isfinite(x), x, 0.0)
        else:
            if not torch.all(torch.isfinite(x)):
                logging.warning("Output contains non-finite values. Replacing with zeros.")
                x = torch.where(torch.isfinite(x), x, torch.zeros_like(x))

        return x

    def _attempt_recovery(self, x: Union[jnp.ndarray, torch.Tensor], error: Exception) -> Union[jnp.ndarray, torch.Tensor]:
        """Attempt to recover from errors during forward pass."""
        logging.info(f"Attempting to recover from error: {str(error)}")

        if isinstance(error, ValueError) and "shape mismatch" in str(error):
            logging.info("Attempting to reshape input.")
            return x.reshape(self.input_shape) if self.backend == 'jax' else x.view(self.input_shape)

        if isinstance(error, (jax.errors.InvalidArgumentError, torch.cuda.OutOfMemoryError)):
            logging.info("Attempting to handle invalid argument or out of memory error.")
            return jnp.clip(x, -1e6, 1e6) if self.backend == 'jax' else torch.clamp(x, -1e6, 1e6)

        logging.info("No specific recovery method. Returning original input.")
        return x

    def _fallback_output(self, x: Union[jnp.ndarray, torch.Tensor]) -> Union[jnp.ndarray, torch.Tensor]:
        """Generate a fallback output in case of persistent errors."""
        logging.warning("Generating fallback output.")
        return jnp.zeros(self.output_shape, dtype=x.dtype) if self.backend == 'jax' else torch.zeros(self.output_shape, dtype=x.dtype)

    def cnn_block(self, x: Union[jnp.ndarray, torch.Tensor], deterministic: bool) -> Union[jnp.ndarray, torch.Tensor]:
        """Apply CNN layers to the input."""
        if self.backend == 'jax':
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = conv(x)
                x = bn(x, use_running_average=deterministic)
                x = self.activation(x)
                x = nn.max_pool(x, window_shape=(2,) * self.conv_dim, strides=(2,) * self.conv_dim)
            return x.reshape((x.shape[0], -1))  # Flatten
        else:
            for conv, bn in zip(self.conv_layers, self.bn_layers):
                x = conv(x)
                x = bn(x)
                x = self.activation(x)
                x = torch.nn.functional.max_pool2d(x, 2)
            return torch.flatten(x, 1)

    def dnn_block(self, x: Union[jnp.ndarray, torch.Tensor], deterministic: bool) -> Union[jnp.ndarray, torch.Tensor]:
        """Apply DNN layers to the input."""
        if self.backend == 'jax':
            for layer in self.dense_layers[:-1]:
                x = layer(x)
                x = self.activation(x)
            x = self.dense_layers[-1](x, deterministic=deterministic)  # Dropout layer
        else:
            for layer in self.dense_layers[:-1]:
                x = self.activation(layer(x))
            x = self.dense_layers[-1](x)
        return x

    def _validate_input_shape(self, x: Union[jnp.ndarray, torch.Tensor]) -> None:
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
            return int(jnp.prod(jnp.array(cnn_output_shape))) if self.backend == 'jax' else int(torch.prod(torch.tensor(cnn_output_shape)))
        else:
            return int(jnp.prod(jnp.array(self.input_shape[1:]))) if self.backend == 'jax' else int(torch.prod(torch.tensor(self.input_shape[1:])))

def create_neuroflex_nn(features: Sequence[int], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], use_cnn: bool = False, conv_dim: int = 2, use_rl: bool = False, action_dim: Optional[int] = None, dtype: Any = jnp.float32, backend: str = 'jax', num_levels: int = 1, sub_action_dims: Optional[List[int]] = None) -> NeuroFlexNN:
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
        dtype (Any, optional): The data type to use for computations. Defaults to jnp.float32.
        backend (str, optional): The backend to use ('jax' or 'torch'). Defaults to 'jax'.
        num_levels (int, optional): Number of levels for hierarchical RL. Defaults to 1.
        sub_action_dims (Optional[List[int]], optional): Action dimensions for each level in hierarchical RL. Defaults to None.

    Returns:
        NeuroFlexNN: An instance of the NeuroFlexNN class.

    Example:
        >>> model = create_neuroflex_nn([64, 32, 10], input_shape=(1, 28, 28, 1), output_shape=(1, 10), use_cnn=True, backend='torch', num_levels=2, sub_action_dims=[4, 2])
    """
    return NeuroFlexNN(features=features, input_shape=input_shape, output_shape=output_shape,
                       use_cnn=use_cnn, conv_dim=conv_dim, use_rl=use_rl, action_dim=action_dim,
                       dtype=dtype, backend=backend, num_levels=num_levels, sub_action_dims=sub_action_dims)

# Advanced neural network components including RL
class AdvancedNNComponents:
    def __init__(self, backend: str = 'jax'):
        self.replay_buffer = None
        self.optimizer = None
        self.epsilon = None
        self.backend = backend

    def initialize_rl_components(self, buffer_size: int, learning_rate: float, epsilon_start: float):
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        self.optimizer = optax.adam(learning_rate) if self.backend == 'jax' else torch.optim.Adam([], lr=learning_rate)
        self.epsilon = epsilon_start

    def update_rl_model(self, state, target_state, batch):
        if self.backend == 'jax':
            def loss_fn(params):
                q_values = state.apply_fn({'params': params}, batch['observations'])
                next_q_values = target_state.apply_fn({'params': target_state.params}, batch['next_observations'])
                targets = batch['rewards'] + self.gamma * jnp.max(next_q_values, axis=-1) * (1 - batch['dones'])
                loss = jnp.mean(optax.huber_loss(q_values[jnp.arange(len(batch['actions'])), batch['actions']], targets))
                return loss

            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            state = state.apply_gradients(grads=grads)
        else:
            q_values = state(batch['observations'])
            with torch.no_grad():
                next_q_values = target_state(batch['next_observations'])
            targets = batch['rewards'] + self.gamma * torch.max(next_q_values, dim=-1)[0] * (1 - batch['dones'])
            loss = torch.nn.functional.smooth_l1_loss(q_values.gather(1, batch['actions'].unsqueeze(-1)).squeeze(-1), targets)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            state = state  # In PyTorch, the model is updated in-place

        return state, loss

    def select_action(self, state, observation, epsilon):
        if self.backend == 'jax':
            if jax.random.uniform(jax.random.PRNGKey(0)) < epsilon:
                return jax.random.randint(jax.random.PRNGKey(0), (), 0, state.output_dim)
            else:
                q_values = state.apply_fn({'params': state.params}, observation[None, ...])
                return jnp.argmax(q_values[0])
        else:
            if torch.rand(1).item() < epsilon:
                return torch.randint(0, state.output_dim, (1,)).item()
            else:
                with torch.no_grad():
                    q_values = state(observation.unsqueeze(0))
                return torch.argmax(q_values[0]).item()

def create_rl_train_state(rng, model, dummy_input, optimizer):
    if isinstance(model, nn.Module):  # JAX model
        params = model.init(rng, dummy_input)['params']
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
    else:  # PyTorch model
        return model  # PyTorch models don't use train_state
