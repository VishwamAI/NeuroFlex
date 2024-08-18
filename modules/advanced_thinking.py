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
                                for feat in features[:-1]]  # Include all features except the last
            self.dense_layers = [nn.Dense(features[-2], dtype=self.dtype)]  # Add one dense layer after conv layers
        else:
            self.dense_layers = [nn.Dense(feat, dtype=self.dtype) for feat in features[:-1]]
        self.final_layer = nn.Dense(features[-1], dtype=self.dtype)

    def _init_rl_agent(self):
        if self.action_dim is None:
            raise ValueError("action_dim must be specified for RL agent initialization")
        rl_features = self.features[:-1]
        self.rl_agent = RLAgent(features=rl_features, action_dim=self.action_dim)
        logging.info(f"Initialized RL agent with features {rl_features} and action_dim {self.action_dim}")
        self.rl_initialized = True

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = self._preprocess_input(x)
        self._validate_input(x)

        if self.use_cnn:
            x = self.cnn_block(x)
        x = self.dnn_block(x)

        if self.use_rl:
            if not hasattr(self, 'rl_agent'):
                raise ValueError("RL agent not initialized. Call _init_rl_agent first.")
            x = self.rl_agent(x)
        else:
            x = self.final_layer(x)

        # Ensure output has correct shape
        expected_output_dim = self.action_dim if self.use_rl else self.output_shape
        if x.shape[-1] != expected_output_dim:
            raise ValueError(f"Output dimension mismatch. Expected {expected_output_dim}, got {x.shape[-1]}")

        return x

    def _preprocess_input(self, x: jnp.ndarray) -> jnp.ndarray:
        if x.ndim == 1:
            x = x.reshape(1, -1)
        elif self.use_cnn and x.ndim == self.conv_dim + 1:
            x = x.reshape(1, *x.shape)
        return x

    def cnn_block(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.conv_dim not in [2, 3]:
            raise ValueError(f"Invalid conv_dim: {self.conv_dim}. Must be 2 or 3.")

        pool_shape = (2, 2) if self.conv_dim == 2 else (2, 2, 2)

        for layer in self.conv_layers:
            x = self.activation(layer(x))
            x = nn.max_pool(x, window_shape=pool_shape, strides=pool_shape)

        return x.reshape((x.shape[0], -1))  # Flatten the output

    def dnn_block(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.dense_layers:
            x = self.activation(layer(x))
        return x

    def _validate_input(self, x: jnp.ndarray):
        if self.use_cnn:
            expected_dim = self.conv_dim + 2  # batch_size, height, width, (depth), channels
            if x.ndim != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim} for CNN, got {x.ndim}. "
                                 f"Input shape: {x.shape}")
            if self.conv_dim == 2:
                if x.shape[-1] not in [1, 3]:
                    raise ValueError(f"Expected 1 or 3 channels for 2D CNN, got {x.shape[-1]}. "
                                     f"Input shape: {x.shape}")
            elif self.conv_dim == 3:
                if x.shape[-1] != 1:
                    raise ValueError(f"Expected 1 channel for 3D CNN, got {x.shape[-1]}. "
                                     f"Input shape: {x.shape}")
        else:
            if x.ndim != 2:
                raise ValueError(f"Expected 2D input for DNN, got {x.ndim}D. Input shape: {x.shape}")
            if x.shape[1] != self.features[0]:
                raise ValueError(f"Input feature size {x.shape[1]} does not match first layer size {self.features[0]}. "
                                 f"Input shape: {x.shape}, Expected features: {self.features}")

        # Check for NaN or infinite values
        if not jnp.all(jnp.isfinite(x)):
            raise ValueError("Input contains NaN or infinite values")

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
        variables = model.init(rng, dummy_input)
        params = jax.tree_map(lambda x: x.astype(jnp.float32), variables['params'])

        # Ensure all layers are properly initialized
        layer_types = ['conv_layers', 'dense_layers', 'final_layer']
        for layer_type in layer_types:
            if hasattr(model, layer_type):
                layers = getattr(model, layer_type)
                if isinstance(layers, list):
                    for i, layer in enumerate(layers):
                        layer_name = f"{layer_type}_{i}"
                        if layer_name not in params:
                            try:
                                if layer_type == 'conv_layers':
                                    # For CNN layers, we need to use the appropriate input shape
                                    if i == 0:
                                        layer_input = dummy_input
                                    else:
                                        prev_layer_output = model.conv_layers[i-1](layer_input)
                                        layer_input = prev_layer_output
                                else:
                                    layer_input = dummy_input
                                layer_params = layer.init(rng, layer_input)['params']
                                params[layer_name] = jax.tree_map(lambda x: x.astype(jnp.float32), layer_params)
                                logging.info(f"Initialized {layer_name}")
                            except Exception as layer_error:
                                logging.error(f"Error initializing {layer_name}: {str(layer_error)}")
                                raise ValueError(f"Failed to initialize {layer_name}: {str(layer_error)}")
                else:
                    if layer_type not in params:
                        try:
                            layer_params = layers.init(rng, dummy_input)['params']
                            params[layer_type] = jax.tree_map(lambda x: x.astype(jnp.float32), layer_params)
                            logging.info(f"Initialized {layer_type}")
                        except Exception as layer_error:
                            logging.error(f"Error initializing {layer_type}: {str(layer_error)}")
                            raise ValueError(f"Failed to initialize {layer_type}: {str(layer_error)}")

        # Handle RL-specific initialization
        if model.use_rl:
            if not hasattr(model, 'rl_agent'):
                raise ValueError("RL agent not found in the model")
            if 'rl_agent' not in params:
                try:
                    rl_dummy_input = dummy_input
                    if model.use_cnn:
                        # Use the output of the CNN block as input for the RL agent
                        cnn_output = model.cnn_block(dummy_input)
                        rl_dummy_input = cnn_output
                    rl_params = model.rl_agent.init(rng, rl_dummy_input)['params']
                    params['rl_agent'] = jax.tree_map(lambda x: x.astype(jnp.float32), rl_params)
                    logging.info("Initialized RL agent")
                except Exception as rl_error:
                    logging.error(f"Error initializing RL agent: {str(rl_error)}")
                    raise ValueError(f"Failed to initialize RL agent: {str(rl_error)}")

        # Verify input shape compatibility
        try:
            _ = model.apply({'params': params}, dummy_input)
        except Exception as shape_error:
            logging.error(f"Input shape incompatibility: {str(shape_error)}")
            raise ValueError(f"Model initialization failed due to input shape mismatch: {str(shape_error)}")

        schedule = optax.exponential_decay(init_value=learning_rate,
                                           transition_steps=1000,
                                           decay_rate=0.9)
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),  # Gradient clipping
            optax.adam(learning_rate=schedule)
        )

        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        logging.info("Successfully created train state")
        return state

    except Exception as e:
        logging.error(f"Error creating train state: {str(e)}")
        raise ValueError(f"Failed to create train state: {str(e)}")

    finally:
        # Log the final structure of the model for debugging
        logging.debug(f"Final model structure: {jax.tree_map(lambda x: x.shape, params)}")

@jit
def select_action(observation, model, params):
    try:
        # Ensure observation is always 2D
        if observation.ndim == 1:
            observation = observation.reshape(1, -1)

        if model.use_rl:
            if 'rl_agent' not in params:
                raise ValueError("RL agent parameters not found")
            action_values = model.rl_agent.apply({'params': params['rl_agent']}, observation)
        else:
            action_values = model.apply({'params': params}, observation)

        # Ensure action_values is always 2D
        if action_values.ndim == 1:
            action_values = action_values.reshape(1, -1)

        # Use jnp.argmax with a specified axis for consistency
        actions = jnp.argmax(action_values, axis=-1)

        # Return scalar for single observations, array for batches
        return jnp.squeeze(actions) if observation.shape[0] == 1 else actions

    except Exception as e:
        logging.error(f"Error in select_action: {str(e)}")
        raise

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
