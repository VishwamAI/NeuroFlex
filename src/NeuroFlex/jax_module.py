# JAX specific implementations will go here

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn
import optax
from typing import Any, Tuple, List, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)

# Flexible model using JAX
class JAXModel(nn.Module):
    features: List[int]
    use_cnn: bool = False
    conv_dim: int = 2
    dtype: jnp.dtype = jnp.float32
    activation: Callable = nn.relu

    def setup(self):
        if self.use_cnn:
            if self.conv_dim not in [2, 3]:
                raise ValueError(f"Invalid conv_dim: {self.conv_dim}. Must be 2 or 3.")
            kernel_size = (3, 3) if self.conv_dim == 2 else (3, 3, 3)
            self.conv_layers = [nn.Conv(features=feat, kernel_size=kernel_size, padding='SAME', dtype=self.dtype)
                                for feat in self.features[:-1]]
        self.dense_layers = [nn.Dense(feat, dtype=self.dtype) for feat in self.features[:-1]]
        self.final_layer = nn.Dense(self.features[-1], dtype=self.dtype)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.use_cnn:
            expected_dim = self.conv_dim + 2  # batch_size, height, width, (depth), channels
            if len(x.shape) != expected_dim:
                raise ValueError(f"Expected input dimension {expected_dim}, got {len(x.shape)}")
            for layer in self.conv_layers:
                x = self.activation(layer(x))
                x = nn.max_pool(x, window_shape=(2,) * self.conv_dim, strides=(2,) * self.conv_dim)
            x = x.reshape((x.shape[0], -1))  # Flatten the output
        else:
            if len(x.shape) != 2:
                raise ValueError(f"Expected 2D input for DNN, got {len(x.shape)}D")
        for layer in self.dense_layers:
            x = self.activation(layer(x))
        return self.final_layer(x)

# JAX-based training function with flexible loss and optimizer
def train_jax_model(
    model: JAXModel,
    params: Any,
    X: jnp.ndarray,
    y: jnp.ndarray,
    loss_fn: Callable = lambda pred, y: jnp.mean((pred - y) ** 2),
    epochs: int = 100,
    patience: int = 20,
    min_delta: float = 1e-6,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    grad_clip_value: float = 1.0
) -> Tuple[Any, float, List[float]]:
    num_samples = X.shape[0]
    num_batches = max(1, int(np.ceil(num_samples / batch_size)))
    total_steps = epochs * num_batches

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.1,
        peak_value=learning_rate,
        warmup_steps=min(100, total_steps // 10),
        decay_steps=total_steps,
        end_value=learning_rate * 0.01
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_value),
        optax.adam(lr_schedule)
    )
    opt_state = optimizer.init(params)

    @jax.jit
    def update(params: Any, opt_state: Any, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[Any, Any, float, Any]:
        def loss_wrapper(params):
            pred = model.apply({'params': params}, x)
            return loss_fn(pred, y)
        loss, grads = jax.value_and_grad(loss_wrapper)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss, grads

    best_loss = float('inf')
    best_params = params
    patience_counter = 0
    training_history = []
    plateau_threshold = 1e-8
    plateau_count = 0
    max_plateau_count = 15

    try:
        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(num_batches):
                try:
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, num_samples)
                    batch_X = X[start_idx:end_idx]
                    batch_y = y[start_idx:end_idx]

                    # Ensure batch_X and batch_y have consistent shapes
                    if batch_X.shape[0] != batch_y.shape[0]:
                        min_size = min(batch_X.shape[0], batch_y.shape[0])
                        batch_X = batch_X[:min_size]
                        batch_y = batch_y[:min_size]

                    params, opt_state, batch_loss, grads = update(params, opt_state, batch_X, batch_y)

                    if not jnp.isfinite(batch_loss):
                        logging.warning(f"Non-finite loss detected: {batch_loss}. Skipping this batch.")
                        continue

                    epoch_loss += batch_loss

                except jax.errors.InvalidArgumentError as iae:
                    logging.warning(f"JAX InvalidArgumentError in batch {i}: {str(iae)}. Skipping this batch.")
                    continue
                except Exception as e:
                    logging.warning(f"Unexpected error in batch {i}: {str(e)}. Skipping this batch.")
                    continue

            if num_batches > 0:
                avg_epoch_loss = epoch_loss / num_batches
            else:
                logging.warning("No valid batches in this epoch.")
                continue

            training_history.append(avg_epoch_loss)

            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_epoch_loss:.6f}")

            if avg_epoch_loss < best_loss - min_delta:
                best_loss = avg_epoch_loss
                best_params = jax.tree_map(lambda x: x.copy(), params)  # Create a copy of the best params
                patience_counter = 0
                plateau_count = 0
                logging.info(f"New best loss: {best_loss:.6f}")
            else:
                patience_counter += 1
                if abs(avg_epoch_loss - best_loss) < plateau_threshold:
                    plateau_count += 1
                    logging.info(f"Plateau detected. Count: {plateau_count}")
                else:
                    plateau_count = 0

            if patience_counter >= patience:
                logging.info(f"Early stopping due to no improvement for {patience} epochs")
                break
            elif plateau_count >= max_plateau_count:
                logging.info(f"Early stopping due to {max_plateau_count} plateaus")
                break

            # Check if loss is decreasing
            if epoch > 0 and avg_epoch_loss > training_history[-2] * 1.1:  # 10% tolerance
                logging.warning(f"Loss increased significantly: {training_history[-2]:.6f} -> {avg_epoch_loss:.6f}")
                # Implement learning rate reduction on significant loss increase
                current_lr = lr_schedule(epoch * num_batches)
                new_lr = current_lr * 0.5
                lr_schedule = optax.exponential_decay(
                    init_value=new_lr,
                    transition_steps=num_batches,
                    decay_rate=0.99
                )
                optimizer = optax.chain(
                    optax.clip_by_global_norm(grad_clip_value),
                    optax.adam(lr_schedule)
                )
                opt_state = optimizer.init(params)
                logging.info(f"Reduced learning rate to {new_lr:.6f}")

            # Monitor gradient norms
            grad_norm = optax.global_norm(jax.tree_map(lambda x: x.astype(jnp.float32), grads))
            logging.info(f"Gradient norm: {grad_norm:.6f}")

            # Implement gradient noise addition
            if grad_norm < 1e-6:
                noise_scale = 1e-6
                noisy_grads = jax.tree_map(lambda x: x + jax.random.normal(jax.random.PRNGKey(epoch), x.shape) * noise_scale, grads)
                updates, opt_state = optimizer.update(noisy_grads, opt_state)
                params = optax.apply_updates(params, updates)
                logging.info("Added gradient noise due to small gradient norm")

    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}")
        raise

    # Ensure consistent parameter shapes
    best_params = jax.tree_map(lambda x: x.astype(jnp.float32), best_params)

    logging.info(f"Training completed. Best loss: {best_loss:.6f}")
    return best_params, best_loss, training_history

# Improved batch prediction with better error handling
@jax.jit
def batch_predict(params: Any, x: jnp.ndarray, use_cnn: bool = False, conv_dim: int = 2) -> jnp.ndarray:
    try:
        # Validate params structure
        if not isinstance(params, dict):
            raise ValueError("params must be a dictionary")

        # Determine the number of features dynamically
        layer_keys = [k for k in params.keys() if k.startswith(('dense_layers_', 'conv_layers_', 'final_dense'))]
        if not layer_keys:
            raise ValueError("No valid layers found in params")
        last_layer = max(layer_keys, key=lambda k: int(k.split('_')[-1]) if '_' in k else float('inf'))
        num_features = params[last_layer]['kernel'].shape[-1]

        # Dynamically create model based on params structure
        features = [params[k]['kernel'].shape[-1] for k in sorted(layer_keys) if k != 'final_dense']
        features.append(num_features)
        model = JAXModel(features=features, use_cnn=use_cnn, conv_dim=conv_dim)

        # Ensure input is a JAX array and handle different input shapes
        if not isinstance(x, jnp.ndarray):
            x = jnp.array(x)
        original_shape = x.shape
        if use_cnn:
            expected_dims = conv_dim + 2  # batch, height, width, (depth), channels
            if x.ndim == expected_dims - 1:
                x = x.reshape(1, *x.shape)  # Add batch dimension for single image
            elif x.ndim != expected_dims:
                raise ValueError(f"Invalid input shape for CNN. Expected {expected_dims} dimensions, got {x.ndim}. Input shape: {original_shape}")
        else:
            if x.ndim == 1:
                x = x.reshape(1, -1)
            elif x.ndim == 0:
                x = x.reshape(1, 1)
            elif x.ndim != 2:
                raise ValueError(f"Invalid input shape for DNN. Expected 2 dimensions, got {x.ndim}. Input shape: {original_shape}")

        # Ensure x has the correct input dimension
        first_layer_key = min(layer_keys, key=lambda k: int(k.split('_')[-1]) if '_' in k else float('inf'))
        expected_input_dim = params[first_layer_key]['kernel'].shape[0]
        if not use_cnn and x.shape[-1] != expected_input_dim:
            raise ValueError(f"Input dimension mismatch. Expected {expected_input_dim}, got {x.shape[-1]}. Input shape: {original_shape}")

        # Apply the model
        output = model.apply({'params': params}, x)

        # Reshape output to match input shape if necessary
        if len(original_shape) > 2 and not use_cnn:
            output = output.reshape(original_shape[:-1] + (-1,))
        elif len(original_shape) == 0:
            output = output.squeeze()

        logging.info(f"Batch prediction successful. Input shape: {original_shape}, Output shape: {output.shape}")
        return output
    except ValueError as ve:
        logging.error(f"ValueError in batch_predict: {str(ve)}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in batch_predict: {str(e)}")
        raise RuntimeError(f"Batch prediction failed: {str(e)}")

# Example of using pmap for multi-device computation
@jax.pmap
def parallel_train(model: JAXModel, params: Any, x: jnp.ndarray, y: jnp.ndarray) -> Tuple[Any, float]:
    return train_jax_model(model, params, x, y)
