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
import tensorflow as tf
import numpy as np
from .rl_module import PrioritizedReplayBuffer, create_train_state, select_action, RLAgent
from .tensorflow_convolutions import TensorFlowConvolutions
from .utils import create_backend, convert_array, get_activation_function


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
        backend (str, optional): The backend to use ('jax' or 'tensorflow'). Defaults to 'jax'.
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
    backend: str = 'jax'

    def setup(self):
        """Initialize the layers of the neural network."""
        logging.info("Starting NeuroFlexNN setup")

        try:
            self._validate_shapes()
            logging.info("Shape validation completed successfully")
        except ValueError as e:
            logging.error(f"Shape validation failed: {str(e)}")
            self._cleanup()
            raise

        self.step_count = self.variable('state', 'step_count', jnp.zeros, shape=(), dtype=jnp.int32)
        logging.debug(f"Step count initialized: {self.step_count}")

        try:
            self.backend_obj = create_backend(self.backend)
            logging.info(f"Backend '{self.backend}' initialized successfully")

            self._initialize_layers()
            logging.info("Layers initialized successfully")

            self.params = self._initialize_params()
            if self.params is None:
                raise ValueError("Parameters initialization failed")
            logging.info("Parameters initialized successfully")

            if self.use_cnn:
                self.batch_stats = self._initialize_batch_stats()
                if self.batch_stats is None:
                    raise ValueError("Batch stats initialization failed")
                logging.info("Batch stats initialized successfully")

            self._validate_initialization()
            logging.info("Initialization validated successfully")

        except Exception as e:
            logging.error(f"Error during NeuroFlexNN setup: {str(e)}")
            logging.debug(f"Error details: {e}", exc_info=True)
            self._cleanup()
            raise

        try:
            self._verify_params()
            logging.info("Parameters verified successfully")
        except Exception as e:
            logging.error(f"Error during parameter verification: {str(e)}")
            logging.debug(f"Error details: {e}", exc_info=True)
            self._cleanup()
            raise

        self._setup_required_attributes()
        logging.info("NeuroFlexNN setup completed successfully with all required attributes")
        logging.debug(f"Final NeuroFlexNN configuration: {self.__dict__}")

    def _setup_required_attributes(self):
        required_attributes = ['params', 'step_count']
        if self.use_cnn:
            required_attributes.extend(['batch_stats', 'cnn_block'])
        if self.use_rl:
            required_attributes.extend(['rl_layer'] if not self.use_dueling else ['value_stream', 'advantage_stream'])

        for attr in required_attributes:
            if not hasattr(self, attr):
                raise AttributeError(f"Required attribute '{attr}' is missing after setup")
            if getattr(self, attr) is None:
                raise ValueError(f"Required attribute '{attr}' is None after setup")

    def _initialize_params(self):
        """Initialize the parameters of the neural network."""
        logging.info("Starting parameter initialization")
        try:
            rng = self.make_rng('params')
            dummy_input = jnp.ones((1,) + self.input_shape[1:], dtype=self.dtype)
            variables = self.init(rng, dummy_input)
            self.params = variables['params']
            if self.params is None:
                raise ValueError("Parameters initialization failed: params is None")

            logging.info("Parameters initialized. Proceeding with detailed verification.")

            self._log_layer_initialization()
            self._verify_expected_layers()
            self._verify_layer_shapes()
            self._verify_parameter_flow()
            self._verify_custom_layers()

            if self.use_cnn:
                self._verify_cnn_initialization()
            if self.use_rl:
                self._verify_rl_initialization()

            self._verify_finite_params()
            self._perform_dummy_forward_pass(dummy_input)

            logging.info("All parameters successfully initialized and verified")
        except Exception as e:
            logging.error(f"Error during parameter initialization: {str(e)}")
            logging.debug(f"Detailed error information: {e}", exc_info=True)
            self.params = None  # Reset params if initialization fails
            raise
        finally:
            params_status = 'Set' if self.params is not None else 'Not set'
            logging.info(f"Parameter initialization process completed. Params status: {params_status}")
            if self.params is not None:
                logging.debug(f"Final params structure: {jax.tree_map(lambda x: x.shape, self.params)}")

        return self.params

    def _perform_dummy_forward_pass(self, dummy_input):
        """Perform a dummy forward pass to ensure all layers are properly connected."""
        try:
            _ = self.apply({'params': self.params}, dummy_input, train=False)
            logging.info("Dummy forward pass completed successfully")
        except Exception as e:
            logging.error(f"Error during dummy forward pass: {str(e)}")
            raise ValueError("Failed to perform dummy forward pass. Layers may not be properly connected.")

    def _log_layer_initialization(self):
        """Log detailed information about each layer's initialization."""
        for layer_name, layer_params in self.params.items():
            logging.debug(f"Initialized {layer_name}:")
            for param_name, param_value in layer_params.items():
                logging.debug(f"  {param_name} shape: {param_value.shape}")
                if not jnp.all(jnp.isfinite(param_value)):
                    raise ValueError(f"Non-finite values found in {layer_name}.{param_name}")
                logging.debug(f"  {param_name} stats: min={param_value.min():.4f}, max={param_value.max():.4f}, mean={param_value.mean():.4f}, std={param_value.std():.4f}")

    def _verify_expected_layers(self):
        """Verify that all expected layers are present in the initialized parameters."""
        expected_layers = ['conv_layers', 'bn_layers', 'dense_layers'] if self.use_cnn else ['dense_layers']
        if self.use_rl:
            expected_layers.extend(['value_stream', 'advantage_stream'] if self.use_dueling else ['rl_layer'])
        else:
            expected_layers.append('final_dense')

        for layer in expected_layers:
            if layer not in self.params:
                raise ValueError(f"Expected layer '{layer}' not found in initialized parameters")
            logging.debug(f"Verified presence of {layer}")

    def _verify_layer_shapes(self):
        """Verify the shapes of initialized layers."""
        if self.use_cnn:
            for i, feat in enumerate(self.features[:-1]):
                conv_shape = self.params['conv_layers'][f'conv_{i}']['kernel'].shape
                expected_shape = (3,) * self.conv_dim + (self.input_shape[-1] if i == 0 else self.features[i-1], feat)
                if conv_shape != expected_shape:
                    raise ValueError(f"Conv layer {i} shape mismatch. Expected {expected_shape}, got {conv_shape}")

        for i, feat in enumerate(self.features):
            dense_shape = self.params['dense_layers'][f'dense_{i}']['kernel'].shape
            expected_shape = (self.calculate_input_size() if i == 0 else self.features[i-1], feat)
            if dense_shape != expected_shape:
                raise ValueError(f"Dense layer {i} shape mismatch. Expected {expected_shape}, got {dense_shape}")

        logging.debug("All layer shapes verified successfully")

    def _verify_expected_layers(self):
        """Verify that all expected layers are present in the initialized parameters."""
        expected_layers = ['conv_layers', 'bn_layers', 'dense_layers'] if self.use_cnn else ['dense_layers']
        if self.use_rl:
            expected_layers.extend(['value_stream', 'advantage_stream'] if self.use_dueling else ['rl_layer'])
        else:
            expected_layers.append('final_dense')

        for layer in expected_layers:
            if layer not in self.params:
                raise ValueError(f"Expected layer '{layer}' not found in initialized parameters")
            logging.debug(f"Verified presence of {layer}")

    def _verify_parameter_flow(self):
        """Verify that parameters are properly stored and managed."""
        for layer_name, layer_params in self.params.items():
            for param_name, param_value in layer_params.items():
                # Check if parameters are leaf nodes (not further nested)
                if isinstance(param_value, (jnp.ndarray, np.ndarray)):
                    logging.debug(f"Parameter flow verified for {layer_name}.{param_name}")
                else:
                    raise ValueError(f"Unexpected parameter structure in {layer_name}.{param_name}")

        # Verify that params can be accessed and used in a forward pass
        try:
            dummy_input = jnp.ones((1,) + self.input_shape[1:], dtype=self.dtype)
            _ = self.apply({'params': self.params}, dummy_input)
            logging.debug("Successfully performed a forward pass with initialized parameters")
        except Exception as e:
            raise ValueError(f"Error in forward pass with initialized parameters: {str(e)}")

    def _verify_layer_shapes(self):
        """Verify the shapes of initialized layers."""
        if self.use_cnn:
            for i, feat in enumerate(self.features[:-1]):
                conv_shape = self.params['conv_layers'][f'conv_{i}']['kernel'].shape
                expected_shape = (3,) * self.conv_dim + (self.input_shape[-1] if i == 0 else self.features[i-1], feat)
                if conv_shape != expected_shape:
                    raise ValueError(f"Conv layer {i} shape mismatch. Expected {expected_shape}, got {conv_shape}")

        for i, feat in enumerate(self.features):
            dense_shape = self.params['dense_layers'][f'dense_{i}']['kernel'].shape
            expected_shape = (self.calculate_input_size() if i == 0 else self.features[i-1], feat)
            if dense_shape != expected_shape:
                raise ValueError(f"Dense layer {i} shape mismatch. Expected {expected_shape}, got {dense_shape}")

    def _verify_custom_layers(self):
        """Verify the initialization of custom layers."""
        custom_layers = ['ResidualBlock'] if self.use_residual else []
        if self.use_rl and self.use_dueling:
            custom_layers.extend(['value_stream', 'advantage_stream'])

        for layer in custom_layers:
            if layer not in self.params:
                raise ValueError(f"Custom layer '{layer}' not found in initialized parameters")
            logging.debug(f"Verified initialization of custom layer: {layer}")

        if self.use_residual:
            self._verify_residual_blocks()

    def _verify_residual_blocks(self):
        """Verify the initialization of residual blocks."""
        for i, layer in enumerate(self.dense_layers):
            if isinstance(layer, ResidualBlock):
                if f'ResidualBlock_{i}' not in self.params:
                    raise ValueError(f"ResidualBlock_{i} not found in initialized parameters")
                logging.debug(f"Verified initialization of ResidualBlock_{i}")

        for i, feat in enumerate(self.features):
            dense_shape = self.params['dense_layers'][f'dense_{i}']['kernel'].shape
            expected_shape = (self.calculate_input_size() if i == 0 else self.features[i-1], feat)
            if dense_shape != expected_shape:
                raise ValueError(f"Dense layer {i} shape mismatch. Expected {expected_shape}, got {dense_shape}")

        logging.debug("All layer shapes verified successfully")

    def _validate_initialization(self):
        """Validate that all necessary components are initialized."""
        logging.info("Validating initialization")
        if not hasattr(self, 'params'):
            raise ValueError("params attribute is missing")
        if not self.params:
            raise ValueError("params is empty")

        if self.use_cnn:
            if not hasattr(self, 'batch_stats'):
                raise ValueError("batch_stats attribute is missing for CNN")
            if not hasattr(self, 'cnn_block'):
                raise ValueError("cnn_block is missing")
            if 'conv_layers' not in self.params:
                raise ValueError("conv_layers not found in params")

        if not self.use_rl:
            if not hasattr(self, 'final_dense'):
                raise ValueError("final_dense layer is missing for non-RL setup")
        else:
            if 'rl_layer' not in self.params and not (self.use_dueling and 'value_stream' in self.params and 'advantage_stream' in self.params):
                raise ValueError("RL layers not properly initialized")

        logging.info("Initialization validation complete")

    def _initialize_layers(self):
        if self.use_cnn:
            logging.info("Setting up CNN layers")
            self._setup_cnn_layers()
            self.cnn_block = self._create_cnn_block()
            logging.debug(f"CNN layers set up: {self.conv_layers}")
            logging.debug(f"CNN block created: {self.cnn_block}")

        logging.info("Setting up dense layers")
        self._setup_dense_layers()
        logging.debug(f"Dense layers set up: {self.dense_layers}")

        if self.use_rl:
            logging.info("Setting up RL components")
            self._setup_rl_components()
            logging.debug("RL components set up")
        else:
            logging.info("Setting up final dense layer")
            self.final_dense = nn.Dense(
                self.output_shape[-1],
                dtype=self.dtype,
                kernel_init=nn.initializers.kaiming_normal()
            )
            logging.debug(f"Final dense layer set up: {self.final_dense}")

    def _initialize_params(self):
        logging.info("Initializing parameters")
        self.params = self.init_params(self.make_rng('params'))
        if self.params is None:
            raise ValueError("Parameters initialization failed")
        logging.debug(f"Parameters initialized: {jax.tree_map(lambda x: x.shape, self.params)}")

    def _initialize_batch_stats(self):
        if self.use_cnn:
            self.batch_stats = self.init_batch_stats(self.make_rng('batch_stats'))
            if self.batch_stats is None:
                raise ValueError("Batch stats initialization failed")
            logging.debug(f"Batch stats initialized: {jax.tree_map(lambda x: x.shape, self.batch_stats)}")

    def _verify_params(self):
        if not hasattr(self, 'params') or self.params is None:
            raise ValueError("params attribute was not set or is None after initialization")

        logging.info(f"Final params structure: {jax.tree_map(lambda x: x.shape, self.params)}")

        expected_keys = ['conv_layers', 'bn_layers', 'dense_layers'] if self.use_cnn else ['dense_layers']
        if self.use_rl:
            expected_keys.extend(['value_stream', 'advantage_stream'] if self.use_dueling else ['rl_layer'])
        else:
            expected_keys.append('final_dense')

        for key in expected_keys:
            if key not in self.params:
                raise ValueError(f"Expected key '{key}' not found in params")

        logging.info("All expected keys found in params")
    def _validate_initialization(self):
        """Validate that all necessary components are initialized."""
        logging.info("Validating initialization")
        if not hasattr(self, 'params'):
            raise ValueError("params attribute is missing")
        if self.use_cnn and not hasattr(self, 'batch_stats'):
            raise ValueError("batch_stats attribute is missing for CNN")
        if self.use_cnn and not hasattr(self, 'cnn_block'):
            raise ValueError("cnn_block is missing")
        if not self.use_rl and not hasattr(self, 'final_dense'):
            raise ValueError("final_dense layer is missing for non-RL setup")
        logging.info("Initialization validation complete")

    def _cleanup(self):
        """Clean up resources in case of initialization failure."""
        logging.info("Cleaning up resources due to initialization failure")
        # Add cleanup logic here, e.g., resetting attributes
        self.params = None
        self.batch_stats = None
        if hasattr(self, 'cnn_block'):
            del self.cnn_block
        if hasattr(self, 'final_dense'):
            del self.final_dense

    def make_rng(self, name: str) -> jax.random.PRNGKey:
        """Create a new RNG for the given name."""
        return jax.random.PRNGKey(hash(name) % (2**32))

    def _setup_rl_components(self):
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

    def init_params(self, rng):
        """Initialize the parameters of the neural network."""
        dummy_input = jnp.ones((1,) + self.input_shape[1:], dtype=self.dtype)
        variables = self.init(rng, dummy_input)
        self.params = variables['params']
        return self.params

    def _init_params(self, key):
        """Initialize the parameters of the neural network."""
        logging.info("Initializing parameters for NeuroFlexNN")
        params = {}
        dummy_input = jnp.ones((1,) + self.input_shape[1:], dtype=self.dtype)

        if self.use_cnn:
            logging.debug("Initializing CNN layers")
            params['conv_layers'] = {}
            params['bn_layers'] = {}
            for i, feat in enumerate(self.features[:-1]):
                key, subkey = jax.random.split(key)
                kernel_shape = (3,) * self.conv_dim + (dummy_input.shape[-1] if i == 0 else self.features[i-1], feat)
                params['conv_layers'][f'conv_{i}'] = {
                    'kernel': self.param(f'conv_kernel_{i}', nn.initializers.kaiming_normal(), kernel_shape),
                    'bias': self.param(f'conv_bias_{i}', nn.initializers.zeros, (feat,))
                }
                params['bn_layers'][f'bn_{i}'] = {
                    'scale': self.param(f'bn_scale_{i}', nn.initializers.ones, (feat,)),
                    'bias': self.param(f'bn_bias_{i}', nn.initializers.zeros, (feat,))
                }
                logging.debug(f"Initialized CNN layer {i} with shape {kernel_shape}")
            dummy_input = dummy_input.reshape((1, -1))  # Flatten for dense layers

        logging.debug("Initializing dense layers")
        params['dense_layers'] = {}
        for i, feat in enumerate(self.features):
            key, subkey = jax.random.split(key)
            input_dim = dummy_input.shape[-1] if i == 0 else self.features[i-1]
            params['dense_layers'][f'dense_{i}'] = {
                'kernel': self.param(f'dense_kernel_{i}', nn.initializers.kaiming_normal(), (input_dim, feat)),
                'bias': self.param(f'dense_bias_{i}', nn.initializers.zeros, (feat,))
            }
            logging.debug(f"Initialized dense layer {i} with shape ({input_dim}, {feat})")

        if self.use_rl:
            logging.debug("Initializing RL-specific layers")
            if self.use_dueling:
                params['value_stream'] = {
                    'kernel': self.param('value_kernel', nn.initializers.kaiming_normal(), (self.features[-1], 1)),
                    'bias': self.param('value_bias', nn.initializers.zeros, (1,))
                }
                params['advantage_stream'] = {
                    'kernel': self.param('advantage_kernel', nn.initializers.kaiming_normal(), (self.features[-1], self.action_dim)),
                    'bias': self.param('advantage_bias', nn.initializers.zeros, (self.action_dim,))
                }
                logging.debug(f"Initialized dueling streams with shapes: value ({self.features[-1]}, 1), advantage ({self.features[-1]}, {self.action_dim})")
            else:
                params['rl_layer'] = {
                    'kernel': self.param('rl_kernel', nn.initializers.kaiming_normal(), (self.features[-1], self.action_dim)),
                    'bias': self.param('rl_bias', nn.initializers.zeros, (self.action_dim,))
                }
                logging.debug(f"Initialized RL layer with shape ({self.features[-1]}, {self.action_dim})")
        else:
            logging.debug("Initializing final dense layer")
            params['final_dense'] = {
                'kernel': self.param('final_dense_kernel', nn.initializers.kaiming_normal(), (self.features[-1], self.output_shape[-1])),
                'bias': self.param('final_dense_bias', nn.initializers.zeros, (self.output_shape[-1],))
            }
            logging.debug(f"Initialized final dense layer with shape ({self.features[-1]}, {self.output_shape[-1]})")

        logging.info("Parameter initialization completed")
        return params

    def _create_cnn_block(self):
        """Create the CNN block."""
        def cnn_block(x, train: bool = True, variables: Optional[Dict] = None):
            for i, (conv, bn) in enumerate(zip(self.conv_layers, self.bn_layers)):
                x = conv(x, variables=variables['conv_layers'][f'conv_{i}'] if variables else None)
                x = bn(x, use_running_average=not train, variables=variables['bn_layers'][f'bn_{i}'] if variables else None)
                x = self.activation(x)
                x = nn.max_pool(x, window_shape=(2,) * self.conv_dim, strides=(2,) * self.conv_dim)
            return x.reshape((x.shape[0], -1))  # Flatten
        return cnn_block

    def _fallback_output(self, x: jnp.ndarray) -> jnp.ndarray:
        """Generate a fallback output in case of persistent errors."""
        logging.warning("Generating fallback output.")
        return jnp.zeros(self.output_shape, dtype=self.dtype)

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        """
        Forward pass of the neural network.

        Args:
            x (jnp.ndarray): Input tensor.
            train (bool): Whether to run in training mode. Defaults to True.

        Returns:
            jnp.ndarray: Output tensor.
        """
        logging.debug(f"Input shape: {x.shape}, Train mode: {train}")
        logging.debug(f"Params structure: {jax.tree_map(lambda x: x.shape, self.params)}")

        try:
            self._validate_input_shape(x)
            return self._forward(x, train=train)
        except ValueError as ve:
            logging.warning(f"Input shape mismatch: {str(ve)}")
            x = self._attempt_recovery(x, ve)
        except Exception as e:
            logging.error(f"Error during forward pass: {str(e)}")
            logging.debug(f"Error details: {e}", exc_info=True)
            x = self._attempt_recovery(x, e)

        # If we've reached this point, attempt recovery has been performed
        try:
            return self._forward(x, train=train)
        except Exception as e:
            logging.error(f"Error after recovery attempt: {str(e)}")
            logging.debug(f"Error details: {e}", exc_info=True)
            return self._fallback_output(x)

    def _forward(self, x: jnp.ndarray, train: bool) -> jnp.ndarray:
        """Internal forward pass implementation."""
        return self.apply({'params': self.params}, x, train=train)

    def _validate_shapes(self):
        """Validate the input and output shapes of the network."""
        logging.info("Starting shape validation...")

        def _check_tuple_of_positive_ints(shape, name, min_dims=2):
            if not isinstance(shape, tuple):
                logging.error(f"{name} must be a tuple, got {type(shape)}")
                raise ValueError(f"{name} must be a tuple, got {type(shape)}")
            if len(shape) < min_dims:
                logging.error(f"{name} must have at least {min_dims} dimensions, got {shape}")
                raise ValueError(f"{name} must have at least {min_dims} dimensions, got {shape}")
            if any(not isinstance(dim, int) or dim <= 0 for dim in shape):
                logging.error(f"All dimensions in {name} must be positive integers, got {shape}")
                raise ValueError(f"All dimensions in {name} must be positive integers, got {shape}")
            logging.debug(f"{name} validation passed: {shape}")

        try:
            logging.info(f"Validating input shape: {self.input_shape}")
            _check_tuple_of_positive_ints(self.input_shape, "Input shape")

            logging.info(f"Validating output shape: {self.output_shape}")
            _check_tuple_of_positive_ints(self.output_shape, "Output shape")

            logging.info(f"Validating features: {self.features}")
            if not isinstance(self.features, tuple):
                logging.error(f"Features must be a tuple, got {type(self.features)}")
                raise ValueError(f"Features must be a tuple, got {type(self.features)}")
            if len(self.features) < 2:
                logging.error(f"Features must have at least two elements, got {self.features}")
                raise ValueError(f"Features must have at least two elements, got {self.features}")
            if any(not isinstance(feat, int) or feat <= 0 for feat in self.features):
                logging.error(f"All features must be positive integers, got {self.features}")
                raise ValueError(f"All features must be positive integers, got {self.features}")
            logging.debug(f"Features validation passed: {self.features}")

            logging.info("Validating network architecture...")
            self._validate_network_architecture()

            logging.info("Validating output consistency...")
            self._validate_output_consistency()

            logging.info("Validating layer consistency...")
            self._validate_layer_consistency()

            logging.info("Validating batch size consistency...")
            self._validate_batch_size_consistency()

            logging.info("Validating feature dimensions...")
            self._validate_feature_dimensions()

            logging.info("Validating total parameters...")
            self._validate_total_params()

            if self.use_cnn:
                logging.info("Validating CNN-specific parameters...")
                self._validate_cnn_params()

            if self.use_rl:
                logging.info("Validating RL-specific parameters...")
                self._validate_rl_params()

            # Additional check for input and output shape consistency
            if self.input_shape[0] != self.output_shape[0]:
                logging.error(f"Batch size mismatch. Input shape: {self.input_shape[0]}, Output shape: {self.output_shape[0]}")
                raise ValueError(f"Batch size mismatch. Input shape: {self.input_shape[0]}, Output shape: {self.output_shape[0]}")

            # Check if the last feature matches the output shape
            if self.features[-1] != self.output_shape[-1]:
                logging.error(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")
                raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")

            logging.info("Shape validation completed successfully.")
        except ValueError as e:
            logging.error(f"Shape validation failed: {str(e)}")
            self._log_shape_error_details()
            raise

    def _validate_cnn_params(self):
        if self.conv_dim not in [2, 3]:
            raise ValueError(f"conv_dim must be 2 or 3 for CNN, got {self.conv_dim}")
        if len(self.input_shape) != self.conv_dim + 2:
            raise ValueError(f"For CNN with conv_dim={self.conv_dim}, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
        if self.input_shape[-1] != self.features[0]:
            logging.warning(f"Input channels {self.input_shape[-1]} do not match first feature dimension {self.features[0]}. Adjusting features.")
            self.features = (self.input_shape[-1],) + self.features[1:]
        logging.debug(f"CNN parameters validation passed: conv_dim={self.conv_dim}, input_shape={self.input_shape}, features={self.features}")

    def _validate_rl_params(self):
        if self.action_dim is None:
            raise ValueError("action_dim must be provided when use_rl is True")
        if not isinstance(self.action_dim, int) or self.action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {self.action_dim}")
        if self.action_dim != self.output_shape[-1]:
            raise ValueError(f"action_dim {self.action_dim} must match last dimension of output_shape {self.output_shape[-1]}")
        logging.debug(f"RL parameters validation passed: action_dim={self.action_dim}")

    def _validate_network_architecture(self):
        if self.use_cnn:
            self._validate_cnn_shapes()
        else:
            self._validate_dnn_shapes()

        if self.use_rl:
            self._validate_rl_shapes()

    def _validate_cnn_shapes(self):
        if self.conv_dim not in [2, 3]:
            raise ValueError(f"conv_dim must be 2 or 3 for CNN, got {self.conv_dim}")
        if len(self.input_shape) != self.conv_dim + 2:
            raise ValueError(f"For CNN with conv_dim={self.conv_dim}, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
        if self.input_shape[-1] != self.features[0]:
            raise ValueError(f"For CNN, input channels {self.input_shape[-1]} must match first feature dimension {self.features[0]}")
        logging.debug(f"CNN shape validation passed: conv_dim={self.conv_dim}, input_shape={self.input_shape}")

    def _validate_dnn_shapes(self):
        if len(self.input_shape) != 2:
            raise ValueError(f"For DNN, input shape must have 2 dimensions, got {len(self.input_shape)}")
        if self.input_shape[-1] != self.features[0]:
            raise ValueError(f"For DNN, input features {self.input_shape[-1]} must match first feature dimension {self.features[0]}")
        logging.debug(f"DNN shape validation passed: input_shape={self.input_shape}")

    def _validate_rl_shapes(self):
        if not isinstance(self.action_dim, int) or self.action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer for RL, got {self.action_dim}")
        if self.action_dim != self.output_shape[-1]:
            raise ValueError(f"action_dim {self.action_dim} must match last dimension of output_shape {self.output_shape[-1]}")
        logging.debug(f"RL shape validation passed: action_dim={self.action_dim}")

    def _validate_output_consistency(self):
        if self.features[-1] != self.output_shape[-1]:
            raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")
        logging.debug("Output consistency validation passed.")

    def _validate_layer_consistency(self):
        for i in range(len(self.features) - 1):
            if self.features[i] != self.features[i+1]:
                logging.debug(f"Layer {i} output ({self.features[i]}) matches Layer {i+1} input ({self.features[i+1]})")
            else:
                raise ValueError(f"Layer size mismatch between layer {i} ({self.features[i]}) and layer {i+1} ({self.features[i+1]})")
        logging.debug("Layer consistency validation passed.")

    def _validate_batch_size_consistency(self):
        if self.input_shape[0] != self.output_shape[0]:
            raise ValueError(f"Batch size mismatch. Input shape: {self.input_shape[0]}, Output shape: {self.output_shape[0]}")
        logging.debug("Batch size consistency validation passed.")

    def _validate_feature_dimensions(self):
        for i in range(len(self.features) - 1):
            if self.features[i] > self.features[i+1]:
                logging.warning(f"Feature dimension decreases from layer {i} ({self.features[i]}) to layer {i+1} ({self.features[i+1]})")
        logging.debug("Feature dimensions validation passed.")

    def _validate_total_params(self):
        total_params = self._calculate_total_params()
        if total_params > 1e9:  # Arbitrary large number, adjust as needed
            logging.warning(f"Total parameter count is very large: {total_params}")
        logging.debug(f"Total parameter count validation passed: {total_params}")

    def _calculate_total_params(self):
        """Calculate the total number of parameters in the network."""
        total = 0
        for i in range(len(self.features) - 1):
            if self.use_cnn and i < len(self.features) - 2:
                total += (3 ** self.conv_dim) * self.features[i] * self.features[i+1]  # Conv layers
            else:
                total += self.features[i] * self.features[i+1]  # Dense layers
            total += self.features[i+1]  # Bias terms
        return total

    def _validate_cnn_shapes(self):
        logging.debug("Validating CNN-specific parameters...")
        if not isinstance(self.conv_dim, int) or self.conv_dim not in [2, 3]:
            raise ValueError(f"conv_dim must be 2 or 3, got {self.conv_dim}")
        if len(self.input_shape) != self.conv_dim + 2:
            raise ValueError(f"For CNN with conv_dim={self.conv_dim}, input shape must have {self.conv_dim + 2} dimensions, got {len(self.input_shape)}")
        if self.input_shape[-1] != self.features[0]:
            raise ValueError(f"For CNN, input channels {self.input_shape[-1]} must match first feature dimension {self.features[0]}")

        # Validate CNN output shape
        expected_output_size = self._calculate_cnn_output_size()
        if expected_output_size != self.features[-1]:
            raise ValueError(f"CNN output size mismatch. Expected {expected_output_size}, got {self.features[-1]}")

        logging.debug(f"CNN-specific validations passed. conv_dim: {self.conv_dim}, input_shape: {self.input_shape}")

    def _log_shape_error_details(self):
        """Log detailed information about the current shape configuration."""
        logging.error("Detailed shape configuration:")
        logging.error(f"Input shape: {self.input_shape}")
        logging.error(f"Output shape: {self.output_shape}")
        logging.error(f"Features: {self.features}")
        logging.error(f"Use CNN: {self.use_cnn}")
        logging.error(f"Use RL: {self.use_rl}")
        if self.use_cnn:
            logging.error(f"Conv dim: {self.conv_dim}")
        if self.use_rl:
            logging.error(f"Action dim: {self.action_dim}")

    def _validate_dnn_shapes(self):
        logging.debug("Validating DNN-specific parameters...")
        if len(self.input_shape) != 2:
            raise ValueError(f"For DNN, input shape must have 2 dimensions, got {self.input_shape}")
        if self.input_shape[-1] != self.features[0]:
            raise ValueError(f"For DNN, input features {self.input_shape[-1]} must match first feature dimension {self.features[0]}")
        logging.debug(f"DNN-specific validations passed. input_shape: {self.input_shape}")

    def _validate_rl_shapes(self):
        logging.debug("Validating RL-specific parameters...")
        if self.action_dim is None:
            raise ValueError("action_dim must be provided when use_rl is True")
        if not isinstance(self.action_dim, int) or self.action_dim <= 0:
            raise ValueError(f"action_dim must be a positive integer, got {self.action_dim}")
        if self.action_dim != self.output_shape[-1]:
            raise ValueError(f"action_dim {self.action_dim} must match last dimension of output_shape {self.output_shape[-1]}")
        logging.debug(f"RL-specific validations passed. action_dim: {self.action_dim}")

    def _validate_output_consistency(self):
        if self.features[-1] != self.output_shape[-1]:
            raise ValueError(f"Last feature dimension {self.features[-1]} must match output shape {self.output_shape[-1]}")

        if self.use_cnn:
            expected_output_size = self._calculate_cnn_output_size()
            if expected_output_size != self.output_shape[-1]:
                raise ValueError(f"CNN output size mismatch. Expected {expected_output_size}, got {self.output_shape[-1]}")
        else:
            if self.features[-1] != self.output_shape[-1]:
                raise ValueError(f"DNN output size mismatch. Expected {self.features[-1]}, got {self.output_shape[-1]}")

        logging.debug("Output consistency validation passed.")

    def _validate_layer_consistency(self):
        logging.debug("Validating layer consistency...")
        for i in range(len(self.features) - 1):
            if self.features[i] != self.features[i+1]:
                logging.debug(f"Layer {i} output ({self.features[i]}) matches Layer {i+1} input ({self.features[i+1]})")
            else:
                raise ValueError(f"Layer size mismatch between layer {i} ({self.features[i]}) and layer {i+1} ({self.features[i+1]})")
        logging.debug("Layer consistency validation passed.")

    def _calculate_cnn_output_size(self):
        """Calculate the expected output size for CNN."""
        size = self.input_shape[1:-1]  # Exclude batch size and channels
        for _ in range(len(self.features) - 1):
            size = [s // 2 for s in size]  # Assuming each conv layer halves the size
        return int(jnp.prod(jnp.array(size)) * self.features[-1])

    def _setup_cnn_layers(self):
        if self.backend == 'jax':
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
        elif self.backend == 'tensorflow':
            self.tf_conv = TensorFlowConvolutions(
                features=self.features,
                input_shape=self.input_shape,
                conv_dim=self.conv_dim
            )
            self.tf_model = self.tf_conv.create_model()
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    def _forward(self, x: jnp.ndarray, train: bool, variables: Dict) -> jnp.ndarray:
        """Internal forward pass implementation."""
        params = variables['params']

        if self.use_cnn:
            x = self.cnn_block(x, train, variables={'params': params['cnn_block']})

        for i, layer in enumerate(self.dense_layers):
            x = layer(x, variables={'params': params[f'dense_layers_{i}']})
            x = self.activation(x)

        if self.use_rl:
            if self.use_dueling:
                value = self.value_stream(x, variables={'params': params['value_stream']})
                advantage = self.advantage_stream(x, variables={'params': params['advantage_stream']})
                x = value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))
            else:
                x = self.rl_layer(x, variables={'params': params['rl_layer']})
        else:
            x = self.final_dense(x, variables={'params': params['final_dense']})

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
            x = self.cnn_block(x, train, variables=variables)
            logging.debug(f"After CNN block shape: {x.shape}")
            if x.ndim != 2:
                x = x.reshape((x.shape[0], -1))
                logging.debug(f"Flattened CNN output shape: {x.shape}")

        x = self.dense_layers(x, variables=variables)
        logging.debug(f"After DNN block shape: {x.shape}")

        if self.use_rl:
            if self.use_dueling:
                value = self.value_stream(x, variables=variables)
                advantages = self.advantage_stream(x, variables=variables)
                q_values = value + (advantages - jnp.mean(advantages, axis=-1, keepdims=True))
            else:
                q_values = self.rl_layer(x, variables=variables)

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
            x = self.final_dense(x, variables=variables)
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

def create_neuroflex_nn(features: Sequence[int], input_shape: Tuple[int, ...], output_shape: Tuple[int, ...], use_cnn: bool = False, conv_dim: int = 2, use_rl: bool = False, action_dim: Optional[int] = None, dtype: Union[jnp.dtype, str] = jnp.float32, backend: str = 'jax') -> NeuroFlexNN:
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
        backend (str, optional): The backend to use ('jax' or 'tensorflow'). Defaults to 'jax'.

    Returns:
        NeuroFlexNN: An instance of the NeuroFlexNN class.

    Example:
        >>> model = create_neuroflex_nn([64, 32, 10], input_shape=(1, 28, 28, 1), output_shape=(1, 10), use_cnn=True, backend='tensorflow')
    """
    return NeuroFlexNN(features=features, input_shape=input_shape, output_shape=output_shape,
                       use_cnn=use_cnn, conv_dim=conv_dim, use_rl=use_rl, action_dim=action_dim,
                       dtype=dtype, backend=backend)

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
