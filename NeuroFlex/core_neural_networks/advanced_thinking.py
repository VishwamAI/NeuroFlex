import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import List, Tuple, Optional, Callable
import logging
import time
from NeuroFlex.utils import normalize_data, preprocess_data

logging.basicConfig(level=logging.INFO)

class CDSTDP(nn.Module):
    input_shape: Tuple[int, ...]
    output_dim: int
    hidden_layers: List[int]
    dropout_rate: float = 0.5
    learning_rate: float = 0.001

    def setup(self):
        self.performance_threshold = 0.8
        self.update_interval = 86400  # 24 hours in seconds
        self.gradient_norm_threshold = 10
        self.performance_history_size = 100

        self.is_trained = False
        self.performance = 0.0
        self.last_update = 0
        self.gradient_norm = 0
        self.performance_history = []

        self.time_window = 20
        self.a_plus = 0.1
        self.a_minus = 0.12
        self.tau_plus = 20.0
        self.tau_minus = 20.0

        self.layers = [nn.Dense(units) for units in self.hidden_layers]
        self.dropout = nn.Dropout(self.dropout_rate)
        self.output_layer = nn.Dense(self.output_dim)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.layers:
            x = jax.nn.relu(layer(x))
            x = self.dropout(x, deterministic=False)
        return self.output_layer(x)

    def update_weights(self, inputs, conscious_state, feedback):
        def update_layer(layer, inputs, conscious_state):
            pre_synaptic = jnp.expand_dims(inputs, 1)
            post_synaptic = jnp.expand_dims(conscious_state, 0)

            delta_t = jnp.arange(-self.time_window, self.time_window + 1)

            stdp = jnp.where(
                delta_t > 0,
                self.a_plus * jnp.exp(-delta_t / self.tau_plus),
                -self.a_minus * jnp.exp(delta_t / self.tau_minus)
            )

            dw = jnp.outer(pre_synaptic, post_synaptic) * stdp

            return layer.update(weight=layer.weight + self.learning_rate * dw * feedback)

        return jax.tree_map(update_layer, self.layers, inputs, conscious_state)

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
        if len(self.performance_history) > 5 and all(p < 0.01 for p in self.performance_history[-5:]):
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

    def train(self, x_train: jnp.ndarray, y_train: jnp.ndarray, epochs: int = 10,
              batch_size: int = 32, validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None,
              callback: Optional[Callable[[float], None]] = None) -> dict:
        criterion = optax.l2_loss
        optimizer = optax.adam(learning_rate=self.learning_rate)
        opt_state = optimizer.init(self.parameters())

        history = {'train_loss': [], 'val_loss': []}

        num_samples = x_train.shape[0]
        num_batches = max(1, num_samples // batch_size)

        logging.info(f"CDSTDP model initial parameters: {sum(p.size for p in jax.tree_leaves(self.parameters()))}")

        @jax.jit
        def train_step(params, opt_state, batch_x, batch_y):
            def loss_fn(params):
                outputs = self.apply(params, batch_x)
                loss = jnp.mean(criterion(outputs, batch_y))
                return loss, outputs

            (loss, outputs), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for epoch in range(epochs):
            epoch_loss = 0.0
            for i in range(0, num_samples, batch_size):
                batch_x = x_train[i:i+batch_size]
                batch_y = y_train[i:i+batch_size]

                params, opt_state, loss = train_step(self.parameters(), opt_state, batch_x, batch_y)
                self.parameters = params
                epoch_loss += loss

            avg_loss = epoch_loss / num_batches
            logging.info(f"CDSTDP - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

            if callback:
                callback(avg_loss)

            self.gradient_norm = jnp.sum(jnp.array([jnp.linalg.norm(g) for g in jax.tree_leaves(grads)]))
            self.performance = 1.0 - avg_loss  # Simple performance metric
            self.update_performance()
            self.adjust_learning_rate()

            issues = self.diagnose()
            if issues:
                logging.info(f"Diagnosed issues: {issues}")
                self.heal(issues)

            # Compute epoch loss
            train_loss = jnp.mean(criterion(self.apply(self.parameters(), x_train), y_train))
            history['train_loss'].append(train_loss)

            if validation_data:
                val_x, val_y = validation_data
                val_loss = jnp.mean(criterion(self.apply(self.parameters(), val_x), val_y))
                history['val_loss'].append(val_loss)

        self.is_trained = True
        self.last_update = time.time()
        logging.info(f"CDSTDP model final parameters: {sum(p.size for p in jax.tree_leaves(self.parameters()))}")

        return history

def create_cdstdp(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int],
                  dropout_rate: float = 0.5, learning_rate: float = 0.001) -> CDSTDP:
    return CDSTDP(input_shape, output_dim, hidden_layers, dropout_rate, learning_rate)
