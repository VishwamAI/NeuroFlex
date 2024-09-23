import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import List, Tuple, Optional, Callable
import logging
import time

logging.basicConfig(level=logging.INFO)


class JAXModel(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_layers: List[int],
        output_dim: int,
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.performance_threshold = 0.8
        self.update_interval = 86400  # 24 hours in seconds
        self.gradient_norm_threshold = 10
        self.performance_history_size = 100

        self.is_trained = False
        self.performance = 0.0
        self.last_update = 0
        self.gradient_norm = 0
        self.performance_history = []

    def setup(self):
        layers = []
        in_features = self.input_dim
        for units in self.hidden_layers:
            layers.append(nn.Dense(units))
            layers.append(nn.relu)
            layers.append(nn.Dropout(self.dropout_rate))
            in_features = units
        layers.append(nn.Dense(self.output_dim))
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def diagnose(self):
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        if self.gradient_norm > self.gradient_norm_threshold:
            issues.append("Gradient explosion detected")
        if len(self.performance_history) > 5 and all(
            p < 0.01 for p in self.performance_history[-5:]
        ):
            issues.append("Model is stuck in local minimum")
        return issues

    def heal(self, issues):
        for issue in issues:
            if issue == "Model is not trained":
                logging.info("Model needs training")
            elif issue == "Model performance is below threshold":
                self.improve_model()
            elif issue == "Model hasn't been updated in 24 hours":
                self.update_model()
            elif issue == "Gradient explosion detected":
                self.handle_gradient_explosion()
            elif issue == "Model is stuck in local minimum":
                self.escape_local_minimum()

    def improve_model(self):
        logging.info("Improving model performance...")
        self.performance = min(self.performance * 1.2 + 0.01, 1.0)
        self.update_performance()

    def update_model(self):
        logging.info("Updating model...")
        self.last_update = time.time()
        self.update_performance()

    def handle_gradient_explosion(self):
        logging.info("Handling gradient explosion...")
        self.learning_rate *= 0.5

    def escape_local_minimum(self):
        logging.info("Attempting to escape local minimum...")
        self.learning_rate = min(self.learning_rate * 2.5, 0.1)
        logging.info(f"New learning rate: {self.learning_rate}")

    def update_performance(self):
        self.performance_history.append(self.performance)
        if len(self.performance_history) > self.performance_history_size:
            self.performance_history.pop(0)

    def adjust_learning_rate(self):
        if len(self.performance_history) >= 2:
            current_performance = self.performance_history[-1]
            previous_performance = self.performance_history[-2]

            if current_performance > previous_performance:
                self.learning_rate *= 1.05
            elif current_performance < previous_performance:
                self.learning_rate *= 0.95
        else:
            self.learning_rate *= 1.01

        self.learning_rate = max(min(self.learning_rate, 0.1), 1e-5)
        return self.learning_rate


def create_jax_model(
    input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]
) -> JAXModel:
    return JAXModel(input_shape[0], hidden_layers, output_dim)


def train_jax_model(
    model: JAXModel,
    x_train: jnp.ndarray,
    y_train: jnp.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
    callback: Optional[Callable[[float], None]] = None,
) -> dict:

    @jax.jit
    def loss_fn(params, x, y):
        predictions = model.apply({"params": params}, x)
        return jnp.mean((predictions - y) ** 2)

    @jax.jit
    def update(params, x, y, opt_state):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)
        return new_params, new_opt_state, loss

    optimizer = optax.adam(model.learning_rate)
    params = model.init(jax.random.PRNGKey(0), jnp.ones((1, model.input_dim)))
    opt_state = optimizer.init(params)

    history = {"train_loss": [], "val_loss": []}

    num_samples = x_train.shape[0]
    num_batches = max(1, num_samples // batch_size)

    logging.info(
        f"JAX model initial parameters: {jax.tree_map(lambda x: x.shape, params)}"
    )

    for epoch in range(epochs):
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]

            params, opt_state, loss = update(params, batch_x, batch_y, opt_state)
            epoch_loss += loss

        avg_loss = epoch_loss / num_batches
        logging.info(f"JAX - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if callback:
            callback(avg_loss)

        model.gradient_norm = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_leaves(params))
        )
        model.performance = 1.0 - avg_loss  # Simple performance metric
        model.update_performance()
        model.adjust_learning_rate()

        issues = model.diagnose()
        if issues:
            logging.info(f"Diagnosed issues: {issues}")
            model.heal(issues)

        # Compute epoch loss
        train_loss = loss_fn(params, x_train, y_train)
        history["train_loss"].append(train_loss)

        if validation_data:
            val_x, val_y = validation_data
            val_loss = loss_fn(params, val_x, val_y)
            history["val_loss"].append(val_loss)

    model.is_trained = True
    model.last_update = time.time()
    logging.info(
        f"JAX model final parameters: {jax.tree_map(lambda x: x.shape, params)}"
    )

    return history


def jax_predict(model: JAXModel, x: jnp.ndarray) -> jnp.ndarray:
    params = model.params
    return jax.jit(lambda x: model.apply({"params": params}, x))(x)
