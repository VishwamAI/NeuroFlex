import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Optional
import logging
import time
from ..constants import PERFORMANCE_THRESHOLD, UPDATE_INTERVAL, MAX_HEALING_ATTEMPTS


class WorkingMemory(nn.Module):
    capacity: int
    hidden_size: int

    def setup(self):
        self.memory = self.variable(
            "memory", "buffer", lambda: jnp.zeros((self.capacity, self.hidden_size))
        )
        self.attention = nn.attention.MultiHeadDotProductAttention(num_heads=4)

    def __call__(self, inputs, query):
        attention_output = self.attention(query, self.memory.value, self.memory.value)
        updated_memory = jnp.concatenate([inputs, self.memory.value[:-1]], axis=0)
        self.memory.value = updated_memory
        return attention_output


class ExtendedCognitiveArchitecture(nn.Module):
    num_layers: int
    hidden_size: int
    working_memory_capacity: int
    input_dim: int
    performance_threshold: float = PERFORMANCE_THRESHOLD
    update_interval: int = UPDATE_INTERVAL
    learning_rate: float = 0.001
    max_healing_attempts: int = MAX_HEALING_ATTEMPTS
    attention_heads: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        self.encoder = nn.Sequential(
            [
                nn.Dense(self.hidden_size),
                nn.relu,
                *[nn.Dense(self.hidden_size) for _ in range(self.num_layers - 1)],
                nn.relu,
            ]
        )
        self.working_memory = WorkingMemory(
            self.working_memory_capacity, self.hidden_size
        )
        self.decoder = nn.Sequential(
            [*[nn.Dense(self.hidden_size) for _ in range(self.num_layers)], nn.relu]
        )
        self.output_layer = nn.Dense(1)  # Adjust based on your specific task
        self.attention = nn.MultiHeadDotProductAttention(num_heads=self.attention_heads)
        self.performance = self.variable(
            "metrics", "performance", jnp.float32, lambda: jnp.array(0.0)
        )
        self.last_update_time = self.variable(
            "metrics", "last_update_time", jnp.float32, lambda: jnp.array(time.time())
        )
        self.learning_rate = self.variable(
            "metrics",
            "learning_rate",
            jnp.float32,
            lambda: jnp.array(self.learning_rate),
        )
        self.performance_history = self.variable(
            "metrics", "performance_history", jnp.float32, lambda: jnp.zeros(100)
        )
        self.global_workspace = self.variable(
            "gwt", "global_workspace", jnp.float32, lambda: jnp.zeros(self.hidden_size)
        )

    def __call__(self, inputs, task_context):
        encoded = self.encoder(inputs)
        memory_output = self.working_memory(encoded, task_context)

        # Apply attention mechanism
        attention_output = self.attention(encoded, memory_output, memory_output)

        combined = jnp.concatenate(
            [encoded, attention_output, self.global_workspace.value], axis=-1
        )
        decoded = self.decoder(combined)
        output = self.output_layer(decoded)
        self._update_performance(output)
        self._update_global_workspace(combined)
        return output

    def _update_performance(self, output):
        new_performance = jnp.mean(jnp.abs(output))
        self.performance.value = new_performance
        self.last_update_time.value = jnp.array(time.time())
        self._update_performance_history(new_performance)

    def _update_performance_history(self, new_performance):
        self.performance_history.value = jnp.roll(
            self.performance_history.value, shift=-1
        )
        self.performance_history.value = self.performance_history.value.at[-1].set(
            new_performance
        )

    def _update_global_workspace(self, combined):
        # Update global workspace using attention mechanism
        attention_weights = nn.softmax(nn.Dense(self.hidden_size)(combined))
        self.global_workspace.value = jnp.sum(attention_weights * combined, axis=0)

    def diagnose(self) -> List[str]:
        issues = []
        if self.performance.value < self.performance_threshold:
            issues.append(f"Low performance: {self.performance.value:.4f}")
        if (time.time() - self.last_update_time.value) > self.update_interval:
            issues.append(
                f"Long time since last update: {(time.time() - self.last_update_time.value) / 3600:.2f} hours"
            )
        if len(self.performance_history.value) > 5 and jnp.all(
            self.performance_history.value[-5:] < self.performance_threshold
        ):
            issues.append("Consistently low performance")
        if (
            jnp.isnan(self.global_workspace.value).any()
            or jnp.isinf(self.global_workspace.value).any()
        ):
            issues.append("Global workspace contains NaN or Inf values")
        return issues

    def self_heal(self):
        issues = self.diagnose()
        if issues:
            logging.info(f"Self-healing triggered. Issues: {issues}")
            for attempt in range(self.max_healing_attempts):
                self._adjust_learning_rate()
                self._reinitialize_components()
                new_performance = self._simulate_training()
                if new_performance > self.performance_threshold:
                    logging.info(
                        f"Self-healing successful. New performance: {new_performance:.4f}"
                    )
                    break
                if "Global workspace contains NaN or Inf values" in issues:
                    self._reset_global_workspace()
            else:
                logging.warning("Self-healing unsuccessful after maximum attempts")
                self._apply_drastic_measures()

    def _adjust_learning_rate(self):
        if len(self.performance_history.value) >= 2:
            if self.performance_history.value[-1] > self.performance_history.value[-2]:
                self.learning_rate.value *= 1.05
            else:
                self.learning_rate.value *= 0.95
        self.learning_rate.value = jnp.clip(self.learning_rate.value, 1e-5, 0.1)
        logging.info(f"Adjusted learning rate to {self.learning_rate.value:.6f}")

    def _reinitialize_components(self):
        self.encoder = nn.Sequential(
            nn.Dense(self.hidden_size),
            nn.relu,
            nn.Dropout(self.dropout_rate),
            *[nn.Dense(self.hidden_size) for _ in range(self.num_layers - 1)],
            nn.relu,
            nn.Dropout(self.dropout_rate),
        )
        self.working_memory = WorkingMemory(
            self.working_memory_capacity, self.hidden_size
        )
        self.decoder = nn.Sequential(
            *[nn.Dense(self.hidden_size) for _ in range(self.num_layers)],
            nn.relu,
            nn.Dropout(self.dropout_rate),
        )
        self.output_layer = nn.Dense(1)
        self.attention = nn.MultiHeadDotProductAttention(num_heads=self.attention_heads)
        logging.info(
            "Components reinitialized with dropout for improved regularization."
        )

    def _simulate_training(self):
        dummy_input = jax.random.normal(
            jax.random.PRNGKey(int(time.time())), (1, self.input_dim)
        )
        dummy_task_context = jax.random.normal(
            jax.random.PRNGKey(int(time.time())), (1, self.hidden_size)
        )
        output = self(dummy_input, dummy_task_context)
        return jnp.mean(jnp.abs(output))

    def _reset_global_workspace(self):
        self.global_workspace.value = jnp.zeros(self.hidden_size)
        logging.info("Global workspace reset due to NaN or Inf values.")

    def _apply_drastic_measures(self):
        self.learning_rate.value = 0.001  # Reset to initial learning rate
        self._reinitialize_components()
        self._reset_global_workspace()
        logging.warning(
            "Applied drastic measures: reset learning rate, reinitialized components, and reset global workspace."
        )


class BCIProcessor(nn.Module):
    input_channels: int
    output_size: int

    def setup(self):
        self.feature_extractor = nn.Sequential(
            [
                nn.Conv(features=32, kernel_size=(3, 3)),
                nn.relu,
                nn.Conv(features=64, kernel_size=(3, 3)),
                nn.relu,
                nn.Flatten(),
            ]
        )
        self.classifier = nn.Dense(self.output_size)

    def __call__(self, inputs):
        features = self.feature_extractor(inputs)
        return self.classifier(features)


def create_extended_cognitive_model(
    num_layers: int,
    hidden_size: int,
    working_memory_capacity: int,
    bci_input_channels: int,
    bci_output_size: int,
    input_dim: int,
) -> nn.Module:
    class CombinedModel(nn.Module):
        def setup(self):
            self.cognitive_model = ExtendedCognitiveArchitecture(
                num_layers, hidden_size, working_memory_capacity, input_dim
            )
            self.bci_processor = BCIProcessor(bci_input_channels, bci_output_size)

        def __call__(self, cognitive_input, bci_input, task_context):
            cognitive_output = self.cognitive_model(cognitive_input, task_context)
            bci_output = self.bci_processor(bci_input)
            combined_output = jnp.concatenate([cognitive_output, bci_output], axis=-1)
            self.cognitive_model.self_heal()  # Trigger self-healing after each forward pass
            return combined_output

    return CombinedModel()


# Example usage
if __name__ == "__main__":
    model = create_extended_cognitive_model(
        num_layers=3,
        hidden_size=64,
        working_memory_capacity=10,
        bci_input_channels=32,
        bci_output_size=5,
        input_dim=100,
    )

    # Initialize the model
    key = jax.random.PRNGKey(0)
    cognitive_input = jax.random.normal(key, (1, 100))  # Example cognitive input
    bci_input = jax.random.normal(key, (1, 32, 32, 1))  # Example BCI input
    task_context = jax.random.normal(key, (1, 64))  # Example task context

    params = model.init(key, cognitive_input, bci_input, task_context)

    # Run the model
    output = model.apply(params, cognitive_input, bci_input, task_context)
    print("Model output shape:", output.shape)

    logging.info(
        "Extended Cognitive Architecture model created and tested successfully."
    )
