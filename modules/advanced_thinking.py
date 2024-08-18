import jax
import jax.numpy as jnp
from jax import random, jit
from flax import linen as nn
from flax.training import train_state
import optax
from typing import List, Optional, Callable, Union, Dict, Any, Tuple
from modules.rl_module import RLAgent
import logging

logging.basicConfig(level=logging.INFO)

class NeuroFlexNN(nn.Module):
    features: Union[List[int], Tuple[int, ...]]
    use_cnn: bool = False
    conv_dim: int = 2
    use_rl: bool = False
    output_dim: Optional[int] = None
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu
    action_dim: Optional[int] = None

    def setup(self):
        if not self.features:
            raise ValueError("features list/tuple cannot be empty")
        if self.use_rl and self.action_dim is None:
            raise ValueError("action_dim must be specified when use_rl is True")

        self._build()
        if self.use_rl:
            self._init_rl_agent()

    def _build(self):
        features = list(self.features) if isinstance(self.features, tuple) else self.features
        if self.use_cnn:
            kernel_size = (3, 3) if self.conv_dim == 2 else (3, 3, 3)
            self.conv_layers = [nn.Conv(features=feat, kernel_size=kernel_size, padding='SAME', dtype=self.dtype)
                                for feat in features[:-1]]
            self.dense_layers = [nn.Dense(features[-1], dtype=self.dtype)]
        else:
            self.dense_layers = [nn.Dense(feat, dtype=self.dtype) for feat in features]
        self.final_layer = nn.Dense(self.output_dim or features[-1], dtype=self.dtype)
        logging.info(f"Built NeuroFlexNN with features: {features}, use_cnn: {self.use_cnn}, conv_dim: {self.conv_dim}")

    def _init_rl_agent(self):
        if self.action_dim is None:
            raise ValueError("action_dim must be specified for RL agent initialization")
        rl_features = self.features[:-1]
        self.rl_agent = RLAgent(features=rl_features, action_dim=self.action_dim)
        logging.info(f"Initialized RL agent with features {rl_features} and action_dim {self.action_dim}")
        self.rl_initialized = True

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        try:
            original_shape = x.shape
            logging.debug(f"Original input shape: {original_shape}")

            x = self._preprocess_input(x)
            x = self._validate_input(x)
            logging.debug(f"Input shape after preprocessing and validation: {x.shape}")

            if self.use_cnn:
                x = self.cnn_block(x)
                logging.debug(f"Shape after CNN block: {x.shape}")
            else:
                x = self.dnn_block(x)
                logging.debug(f"Shape after DNN block: {x.shape}")

            if self.use_rl:
                if not hasattr(self, 'rl_agent'):
                    raise AttributeError("RL agent not initialized. Call _init_rl_agent first.")
                x = self.rl_agent(x)
                logging.debug(f"Shape after RL agent: {x.shape}")
            else:
                x = self.final_layer(x)
                logging.debug(f"Shape after final layer: {x.shape}")

            expected_output_dim = self.action_dim if self.use_rl else (self.output_dim or self.features[-1])
            logging.debug(f"Expected output dimension: {expected_output_dim}")

            # Ensure output is always 2D
            if x.ndim == 1:
                x = x.reshape(1, -1)
                logging.debug(f"Reshaped 1D output to 2D: {x.shape}")
            elif x.ndim > 2:
                x = x.reshape(-1, expected_output_dim)
                logging.debug(f"Reshaped >2D output to 2D: {x.shape}")

            # Determine the expected batch size
            expected_batch_size = max(original_shape[0], 1)
            logging.debug(f"Expected batch size: {expected_batch_size}")

            # Ensure consistent output shape for both CNN and DNN cases
            if x.shape[0] != expected_batch_size:
                logging.warning(f"Adjusting batch size from {x.shape[0]} to {expected_batch_size}")
                x = x.reshape(expected_batch_size, -1)

            # Adjust output dimension if necessary
            if x.shape[-1] != expected_output_dim:
                logging.warning(f"Output dimension mismatch. Expected {expected_output_dim}, got {x.shape[-1]}. Adjusting...")
                if x.shape[-1] < expected_output_dim:
                    x = jnp.pad(x, ((0, 0), (0, expected_output_dim - x.shape[-1])))
                else:
                    x = x[..., :expected_output_dim]
                logging.debug(f"Adjusted output shape: {x.shape}")

            # Assertions for shape consistency
            try:
                assert x.ndim == 2, f"Expected 2D output, got {x.ndim}D"
                assert x.shape[0] == expected_batch_size, f"Expected batch size {expected_batch_size}, got {x.shape[0]}"
                assert x.shape[1] == expected_output_dim, f"Expected output dimension {expected_output_dim}, got {x.shape[1]}"
            except AssertionError as ae:
                logging.error(f"Shape consistency check failed: {str(ae)}")
                raise ValueError(f"Output shape is inconsistent: {x.shape}") from ae

            # Check for NaN and Inf values
            if jnp.any(jnp.isnan(x)) or jnp.any(jnp.isinf(x)):
                nan_count = jnp.sum(jnp.isnan(x))
                inf_count = jnp.sum(jnp.isinf(x))
                raise ValueError(f"Output contains {nan_count} NaN and {inf_count} Inf values")

            # Additional check for unexpected large values
            max_abs_value = jnp.max(jnp.abs(x))
            if max_abs_value > 1e5:
                logging.warning(f"Output contains unexpectedly large values: max abs value = {max_abs_value}")

            logging.info(f"Final output shape: {x.shape} (original input shape: {original_shape})")
            return x
        except Exception as e:
            logging.error(f"Error in NeuroFlexNN.__call__: {str(e)}")
            logging.error(f"Original input shape: {original_shape if 'original_shape' in locals() else 'Unknown'}")
            logging.error(f"Current shape: {x.shape if 'x' in locals() else 'Unknown'}")
            logging.error(f"Model configuration: use_cnn={self.use_cnn}, use_rl={self.use_rl}")
            logging.error(f"Expected output dimension: {expected_output_dim if 'expected_output_dim' in locals() else 'Unknown'}")
            raise RuntimeError(f"NeuroFlexNN forward pass failed: {str(e)}") from e

    def _preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        original_shape = x.shape
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif self.use_cnn and x.ndim == self.conv_dim + 1:
            x = x.reshape(1, *x.shape)
        logging.debug(f"Preprocessed input shape: {x.shape} (original: {original_shape})")
        return x

    def cnn_block(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.conv_dim not in [2, 3]:
            raise ValueError(f"Invalid conv_dim: {self.conv_dim}. Must be 2 or 3.")

        logging.debug(f"Input shape to cnn_block: {x.shape}")
        expected_dim = self.conv_dim + 2  # batch_size, height, width, (depth), channels
        if x.ndim != expected_dim:
            raise ValueError(f"Expected {expected_dim}D input, got {x.ndim}D")

        pool_shape = (2, 2) if self.conv_dim == 2 else (2, 2, 2)

        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            logging.debug(f"Shape after conv layer {i} (before activation): {x.shape}")
            x = self.activation(x)
            logging.debug(f"Shape after activation: {x.shape}")
            x = nn.max_pool(x, window_shape=pool_shape, strides=pool_shape)
            logging.debug(f"Shape after max pooling: {x.shape}")

        x = x.reshape((x.shape[0], -1))  # Flatten the output
        logging.debug(f"Shape after flattening: {x.shape}")

        # Apply the dense layer after flattening
        x = self.dense_layers[0](x)
        x = self.activation(x)
        logging.debug(f"Shape after final dense layer: {x.shape}")

        return x

    def dnn_block(self, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(self.dense_layers):
            x = layer(x)
            x = self.activation(x)
            logging.debug(f"Shape after dense layer {i}: {x.shape}")
        return x

    def _validate_input(self, x: jnp.ndarray) -> jnp.ndarray:
        original_shape = x.shape
        logging.debug(f"Input shape: {original_shape}")

        def cnn_validation(x):
            expected_dim = self.conv_dim + 2  # batch_size, height, width, (depth), channels
            logging.debug(f"CNN validation input shape: {x.shape}, expected_dim: {expected_dim}")

            if x.ndim != expected_dim:
                if x.ndim == expected_dim - 1:
                    x = x[None, ...]
                    logging.debug(f"Added batch dimension. New shape: {x.shape}")
                elif x.ndim == 2:
                    # Reshape (batch_size, features) to (batch_size, height, width, channels)
                    batch_size, features = x.shape
                    if self.conv_dim == 2:
                        height = int(jnp.sqrt(features))
                        if height * height == features:
                            x = x.reshape(batch_size, height, height, 1)
                        else:
                            x = x.reshape(batch_size, -1, 1, self.features[0])
                        logging.debug(f"Reshaped 2D input to 4D. New shape: {x.shape}")
                    elif self.conv_dim == 3:
                        depth = int(jnp.cbrt(features))
                        if depth * depth * depth == features:
                            x = x.reshape(batch_size, depth, depth, depth, 1)
                        else:
                            x = x.reshape(batch_size, -1, 1, 1, self.features[0])
                        logging.debug(f"Reshaped 2D input to 5D. New shape: {x.shape}")
                else:
                    raise ValueError(f"Expected {expected_dim} dimensions for CNN input, got {x.ndim}. Shape: {x.shape}")

            if x.shape[-1] != self.features[0]:
                if x.shape[-1] == 1:
                    x = jnp.repeat(x, self.features[0], axis=-1)
                    logging.debug(f"Repeated single channel to match features. New shape: {x.shape}")
                else:
                    logging.warning(f"Input channels mismatch. Expected {self.features[0]}, got {x.shape[-1]}. Adjusting model.")
                    self.features = list(self.features)  # Convert tuple to list if necessary
                    self.features[0] = x.shape[-1]

            logging.debug(f"CNN validation output shape: {x.shape}")
            return x

        def dnn_validation(x):
            logging.debug(f"DNN validation input shape: {x.shape}")
            if x.ndim == 1:
                x = x.reshape(1, -1)
                logging.debug(f"Reshaped 1D input to 2D. New shape: {x.shape}")
            elif x.ndim > 2:
                x = x.reshape(x.shape[0], -1)  # Flatten all but the first dimension
                logging.debug(f"Flattened input to 2D. New shape: {x.shape}")

            if x.shape[1] != self.features[0]:
                logging.warning(f"Input features mismatch. Expected {self.features[0]}, got {x.shape[1]}. Adjusting model.")
                self.features = list(self.features)  # Convert tuple to list if necessary
                self.features[0] = x.shape[1]

            logging.debug(f"DNN validation output shape: {x.shape}")
            return x

        try:
            # Apply appropriate validation
            x = jax.lax.cond(self.use_cnn, cnn_validation, dnn_validation, x)

            def handle_non_finite(x):
                non_finite_mask = ~jnp.isfinite(x)
                if jnp.any(non_finite_mask):
                    logging.warning(f"Non-finite values detected in input. Replacing with 0.")
                    x = jnp.where(non_finite_mask, 0.0, x)
                return x

            x = handle_non_finite(x)
            logging.debug(f"Shape after handling non-finite values: {x.shape}")

            # Ensure output is always 2D
            if x.ndim > 2:
                x = x.reshape(x.shape[0], -1)
                logging.debug(f"Flattened output to 2D. New shape: {x.shape}")
            elif x.ndim == 1:
                x = x.reshape(1, -1)
                logging.debug(f"Reshaped 1D output to 2D. New shape: {x.shape}")

            # Ensure consistent feature dimension
            expected_features = self.features[-1]
            if x.shape[1] != expected_features:
                logging.warning(f"Output feature mismatch. Expected {expected_features}, got {x.shape[1]}. Adjusting output.")
                if x.shape[1] > expected_features:
                    x = x[:, :expected_features]
                else:
                    x = jnp.pad(x, ((0, 0), (0, expected_features - x.shape[1])), mode='constant')
                logging.debug(f"Adjusted output shape: {x.shape}")

            assert x.shape[1] == expected_features, f"Feature dimension mismatch: {x.shape[1]} != {expected_features}"
            logging.info(f"Input validation complete. Original shape: {original_shape}, Final shape: {x.shape}")
            return x
        except Exception as e:
            logging.error(f"Error in _validate_input: {str(e)}")
            logging.error(f"Original input shape: {original_shape}")
            logging.error(f"Model configuration: use_cnn={self.use_cnn}, conv_dim={self.conv_dim}, features={self.features}")
            raise RuntimeError(f"Input validation failed: {str(e)}") from e

    def simulate_consciousness(self, x: jnp.ndarray) -> jnp.ndarray:
        # Placeholder for consciousness simulation
        return x

    @property
    def output_shape(self) -> int:
        return self.features[-1] if self.output_dim is None else self.output_dim

    def apply_activation(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.activation(x)

    def get_output_dim(self) -> int:
        return self.action_dim if self.use_rl else self.output_shape

def create_train_state(rng, model, dummy_input, learning_rate):
    try:
        logging.info(f"Creating train state with dummy input shape: {dummy_input.shape}")

        # Ensure dummy_input is a JAX array
        dummy_input = jnp.asarray(dummy_input)

        # Validate input shape
        if model.use_cnn:
            expected_dim = model.conv_dim + 2  # batch_size, height, width, (depth), channels
            if dummy_input.ndim != expected_dim:
                raise ValueError(f"Expected {expected_dim}D input for CNN, got {dummy_input.ndim}D. Input shape: {dummy_input.shape}")
        elif dummy_input.ndim != 2:
            raise ValueError(f"Expected 2D input for DNN, got {dummy_input.ndim}D. Input shape: {dummy_input.shape}")

        # Ensure dummy_input has a batch dimension
        if dummy_input.ndim == 1:
            dummy_input = dummy_input[None, ...]

        variables = model.init(rng, dummy_input)
        params = jax.tree_map(lambda x: x.astype(jnp.float32), variables['params'])

        # Initialize layers
        layer_types = ['conv_layers', 'dense_layers', 'final_layer']
        for layer_type in layer_types:
            if hasattr(model, layer_type):
                layers = getattr(model, layer_type)
                if isinstance(layers, list):
                    layer_input = dummy_input
                    for i, layer in enumerate(layers):
                        layer_name = f"{layer_type}_{i}"
                        if layer_name not in params:
                            try:
                                if layer_type == 'conv_layers':
                                    if model.use_cnn and i > 0:
                                        layer_input = layers[i-1](layer_input)
                                else:
                                    layer_input = layer_input.reshape(layer_input.shape[0], -1)
                                layer_params = layer.init(rng, layer_input)['params']
                                params[layer_name] = jax.tree_map(lambda x: x.astype(jnp.float32), layer_params)
                                logging.info(f"Initialized {layer_name} with input shape {layer_input.shape}")
                                layer_input = layer(layer_input)
                            except Exception as layer_error:
                                logging.error(f"Error initializing {layer_name}: {str(layer_error)}")
                                raise ValueError(f"Failed to initialize {layer_name}: {str(layer_error)}. Input shape: {layer_input.shape}")
                elif layer_type not in params:
                    try:
                        layer_input = dummy_input.reshape(dummy_input.shape[0], -1) if layer_type == 'dense_layers' else dummy_input
                        layer_params = layers.init(rng, layer_input)['params']
                        params[layer_type] = jax.tree_map(lambda x: x.astype(jnp.float32), layer_params)
                        logging.info(f"Initialized {layer_type} with input shape {layer_input.shape}")
                    except Exception as layer_error:
                        logging.error(f"Error initializing {layer_type}: {str(layer_error)}")
                        raise ValueError(f"Failed to initialize {layer_type}: {str(layer_error)}. Input shape: {layer_input.shape}")

        # Handle RL-specific initialization
        if model.use_rl:
            if not hasattr(model, 'rl_agent'):
                raise ValueError("RL agent not found in the model. Ensure model.use_rl is set correctly.")
            if 'rl_agent' not in params:
                try:
                    rl_dummy_input = dummy_input
                    if model.use_cnn:
                        cnn_output = model.apply({'params': params}, dummy_input, method=model.cnn_block)
                        rl_dummy_input = cnn_output.reshape(cnn_output.shape[0], -1)
                    else:
                        rl_dummy_input = rl_dummy_input.reshape(rl_dummy_input.shape[0], -1)
                    logging.info(f"Initializing RL agent with input shape: {rl_dummy_input.shape}")
                    rl_params = model.rl_agent.init(rng, rl_dummy_input)['params']
                    params['rl_agent'] = jax.tree_map(lambda x: x.astype(jnp.float32), rl_params)
                    logging.info("Initialized RL agent")
                except Exception as rl_error:
                    logging.error(f"Error initializing RL agent: {str(rl_error)}")
                    raise ValueError(f"Failed to initialize RL agent: {str(rl_error)}. Input shape: {rl_dummy_input.shape}")

        # Verify input shape compatibility
        try:
            _ = model.apply({'params': params}, dummy_input)
            logging.info("Successfully verified input shape compatibility")
        except Exception as shape_error:
            logging.error(f"Input shape incompatibility: {str(shape_error)}")
            raise ValueError(f"Model initialization failed due to input shape mismatch: {str(shape_error)}. Input shape: {dummy_input.shape}")

        # Create learning rate schedule and optimizer
        total_steps = 100000
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.1,
            peak_value=learning_rate,
            warmup_steps=min(1000, total_steps // 10),
            decay_steps=total_steps,
            end_value=learning_rate * 0.01
        )
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule)
        )

        # Create and return the train state
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        logging.info("Successfully created train state")

        # Log detailed model structure
        logging.debug("Model structure:")
        _log_model_structure(params)

        # Log optimizer configuration
        logging.debug(f"Optimizer configuration: {tx}")
        logging.debug(f"Initial learning rate: {learning_rate}")

        # Verify RL agent initialization
        if model.use_rl:
            try:
                rl_output = model.rl_agent.apply({'params': params['rl_agent']}, rl_dummy_input)
                logging.info(f"RL agent output shape: {rl_output.shape}")
            except Exception as rl_verify_error:
                logging.error(f"Error verifying RL agent: {str(rl_verify_error)}")
                raise ValueError(f"Failed to verify RL agent: {str(rl_verify_error)}")

        return state

    except Exception as e:
        logging.error(f"Error creating train state: {str(e)}")
        logging.error(f"Model configuration: use_cnn={model.use_cnn}, use_rl={model.use_rl}, features={model.features}")
        raise ValueError(f"Failed to create train state: {str(e)}")

    finally:
        if 'params' in locals():
            logging.debug(f"Final model structure: {jax.tree_map(lambda x: x.shape, params)}")
        else:
            logging.warning("Failed to log final model structure: 'params' not defined")

def _log_model_structure(params, prefix=''):
    for key, value in params.items():
        if isinstance(value, dict):
            _log_model_structure(value, prefix=f"{prefix}{key}.")
        else:
            logging.debug(f"  {prefix}{key}: {value.shape}")

@jit
def select_action(observation, model, params):
    try:
        # Ensure observation is always 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)
        elif observation.ndim > 2:
            raise ValueError(f"Invalid observation shape: {observation.shape}. Expected 1D or 2D array.")

        logging.debug(f"Input observation shape: {observation.shape}")

        if model.use_rl:
            if 'rl_agent' not in params:
                raise ValueError("RL agent parameters not found")
            action_values = model.rl_agent.apply({'params': params['rl_agent']}, observation)
        else:
            action_values = model.apply({'params': params}, observation)

        logging.debug(f"Action values shape before reshape: {action_values.shape}")

        # Ensure action_values is always 2D
        if action_values.ndim == 1:
            action_values = action_values.reshape(1, -1)
        elif action_values.ndim != 2:
            raise ValueError(f"Invalid action_values shape: {action_values.shape}. Expected 1D or 2D array.")

        logging.debug(f"Action values shape after reshape: {action_values.shape}")
        logging.debug(f"Action values: {action_values}")

        # Handle case where all action values are equal
        if jnp.allclose(action_values, action_values[:, :1]):
            logging.warning("All action values are equal. Selecting random action.")
            actions = jax.random.randint(jax.random.PRNGKey(0), shape=action_values.shape[:1], minval=0, maxval=action_values.shape[-1])
        else:
            # Use jnp.argmax with a specified axis for consistency
            actions = jnp.argmax(action_values, axis=-1)

        logging.debug(f"Selected actions shape: {actions.shape}")
        logging.debug(f"Selected actions: {actions}")

        # Return scalar for single observations, array for batches
        return jnp.squeeze(actions) if observation.shape[0] == 1 else actions

    except Exception as e:
        logging.error(f"Error in select_action: {str(e)}")
        logging.error(f"Model use_rl: {model.use_rl}")
        logging.error(f"Params keys: {params.keys()}")
        logging.error(f"Observation shape: {observation.shape}")
        logging.error(f"Action values shape: {action_values.shape if 'action_values' in locals() else 'Not available'}")
        raise

    finally:
        logging.debug(f"Final observation shape: {observation.shape}")
        logging.debug(f"Final action values shape: {action_values.shape if 'action_values' in locals() else 'Not available'}")
        logging.debug(f"Final selected actions shape: {actions.shape if 'actions' in locals() else 'Not available'}")

def data_augmentation(images, key):
    key, subkey1, subkey2, subkey3 = random.split(key, 4)

    # Brightness adjustment
    brightness_factor = random.uniform(subkey1, minval=0.8, maxval=1.2)
    images = jnp.clip(images * brightness_factor, 0, 1)

    # Contrast adjustment
    contrast_factor = random.uniform(subkey2, minval=0.8, maxval=1.2)
    mean = jnp.mean(images, axis=(1, 2, 3), keepdims=True)
    images = jnp.clip((images - mean) * contrast_factor + mean, 0, 1)

    # Random rotation (up to 15 degrees)
    angles = random.uniform(subkey3, minval=-15, maxval=15, shape=(images.shape[0],))

    # Custom rotation function compatible with JAX arrays
    def custom_rotate(img, angle):
        # Convert angle to radians
        angle_rad = jnp.deg2rad(angle)
        cos_angle, sin_angle = jnp.cos(angle_rad), jnp.sin(angle_rad)

        # Create rotation matrix
        rot_matrix = jnp.array([[cos_angle, -sin_angle],
                                [sin_angle, cos_angle]])

        # Get image center
        center = jnp.array(img.shape[:2]) / 2.0

        # Create meshgrid of coordinates
        y, x = jnp.mgrid[:img.shape[0], :img.shape[1]]
        coords = jnp.stack([x - center[1], y - center[0]])

        # Apply rotation
        rotated_coords = jnp.dot(rot_matrix, coords.reshape(2, -1))
        rotated_coords += jnp.expand_dims(center, 1)

        # Reshape coordinates for grid_sample
        grid = rotated_coords.reshape(2, *img.shape[:2]).transpose((1, 2, 0))
        grid = grid[None]  # Add batch dimension

        # Use vmap to apply rotation to each channel separately
        rotated_img = jax.vmap(lambda x: jax.scipy.ndimage.map_coordinates(x, grid[0].T, order=1, mode='constant'))(img.T).T

        return rotated_img.reshape(img.shape)

    # Apply custom rotation to images
    images = jax.vmap(custom_rotate)(images, angles)

    return (images, key)  # Return as a tuple instead of appending

def adversarial_training(model, params, input_data, epsilon, step_size=0.01):
    def loss_fn(params, image, label):
        logits = model.apply({'params': params}, image)
        return jnp.mean(optax.softmax_cross_entropy(logits, label))

    grad_fn = jax.grad(loss_fn, argnums=1)
    image, label = input_data['image'], input_data['label']

    # Compute gradients of the loss with respect to the input image
    grads = grad_fn(params, image, label)

    # Generate adversarial example using Fast Gradient Sign Method (FGSM)
    # with a step size for more controlled perturbations
    perturbation = jnp.clip(step_size * jnp.sign(grads), -epsilon, epsilon)
    perturbed_image = image + perturbation

    # Clip the perturbed image to ensure it's in the valid range [0, 1]
    perturbed_image = jnp.clip(perturbed_image, 0.0, 1.0)

    # Ensure the perturbation is within epsilon range
    final_perturbation = perturbed_image - image
    final_perturbation = jnp.clip(final_perturbation, -epsilon, epsilon)
    final_perturbed_image = image + final_perturbation

    return {'image': final_perturbed_image, 'label': label}

# Add any other necessary functions or classes here
