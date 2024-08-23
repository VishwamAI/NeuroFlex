import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Dict, Any, List, Tuple, Callable
import optax
import logging
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class EthicalAIFramework:
    def __init__(self, model: nn.Module):
        self.model = model
        self.fairness_constraints = {}
        logging.info("EthicalAIFramework initialized")

    def detect_bias(self, data: jnp.ndarray, sensitive_attributes: List[int]) -> Dict[str, float]:
        """
        Detect bias in model predictions based on sensitive attributes.

        Args:
            data (jnp.ndarray): Input data.
            sensitive_attributes (List[int]): Indices of sensitive attributes in the data.

        Returns:
            Dict[str, float]: Dictionary containing bias metrics.
        """
        predictions = self.model.apply({'params': self.model.params}, data)
        bias_metrics = {}

        for attr in sensitive_attributes:
            attr_values = data[:, attr]
            unique_values = jnp.unique(attr_values)

            for value in unique_values:
                mask = attr_values == value
                group_preds = predictions[mask]
                overall_preds = predictions

                # Calculate statistical parity difference
                stat_parity = jnp.mean(group_preds) - jnp.mean(overall_preds)
                bias_metrics[f'stat_parity_{attr}_{value}'] = float(stat_parity)

        logging.info(f"Bias detection completed. Metrics: {bias_metrics}")
        return bias_metrics

    def add_fairness_constraint(self, constraint_fn: Callable, weight: float = 1.0):
        """
        Add a fairness constraint to the model training process.

        Args:
            constraint_fn (Callable): Function that computes the fairness constraint.
            weight (float): Weight of the constraint in the loss function.
        """
        self.fairness_constraints[constraint_fn] = weight
        logging.info(f"Fairness constraint added with weight {weight}")

    def train_with_fairness(self, train_data: jnp.ndarray, labels: jnp.ndarray,
                            learning_rate: float = 1e-3, num_epochs: int = 100):
        """
        Train the model with fairness constraints.

        Args:
            train_data (jnp.ndarray): Training data.
            labels (jnp.ndarray): Training labels.
            learning_rate (float): Learning rate for optimization.
            num_epochs (int): Number of training epochs.
        """
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(self.model.params)

        @jax.jit
        def loss_fn(params, x, y):
            predictions = self.model.apply({'params': params}, x)
            main_loss = optax.softmax_cross_entropy(predictions, y).mean()
            fairness_loss = sum(weight * constraint_fn(params, x, y)
                                for constraint_fn, weight in self.fairness_constraints.items())
            return main_loss + fairness_loss

        @jax.jit
        def train_step(params, opt_state, x, y):
            loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss

        for epoch in range(num_epochs):
            self.model.params, opt_state, loss = train_step(self.model.params, opt_state, train_data, labels)
            if epoch % 10 == 0:
                logging.info(f"Epoch {epoch}, Loss: {loss}")

        logging.info("Training with fairness constraints completed")

    def generate_transparency_report(self, test_data: jnp.ndarray, test_labels: jnp.ndarray) -> Dict[str, Any]:
        """
        Generate a transparency report for model decisions.

        Args:
            test_data (jnp.ndarray): Test data.
            test_labels (jnp.ndarray): True labels for test data.

        Returns:
            Dict[str, Any]: Dictionary containing transparency metrics and visualizations.
        """
        predictions = self.model.apply({'params': self.model.params}, test_data)
        pred_labels = jnp.argmax(predictions, axis=1)

        # Compute confusion matrix
        cm = confusion_matrix(test_labels, pred_labels)

        # Visualize confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png')
        plt.close()

        # Feature importance (assuming a simple linear model for demonstration)
        feature_importance = jnp.abs(self.model.params['Dense_0']['kernel']).mean(axis=1)

        # Visualize feature importance
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(feature_importance)), feature_importance)
        plt.title('Feature Importance')
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.savefig('feature_importance.png')
        plt.close()

        report = {
            'confusion_matrix': cm.tolist(),
            'feature_importance': feature_importance.tolist(),
            'accuracy': (pred_labels == test_labels).mean().item()
        }

        logging.info("Transparency report generated")
        return report

def create_ethical_ai_framework(model: nn.Module) -> EthicalAIFramework:
    """
    Create an instance of the EthicalAIFramework.

    Args:
        model (nn.Module): The model to be used with the ethical AI framework.

    Returns:
        EthicalAIFramework: An instance of the EthicalAIFramework.
    """
    return EthicalAIFramework(model)

# Example usage
if __name__ == "__main__":
    # Create a simple model
    class SimpleModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(10)(x)
            return nn.softmax(x)

    # Initialize the model
    model = SimpleModel()
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 5)))['params']
    model = model.bind({'params': params})

    # Create ethical AI framework
    ethical_framework = create_ethical_ai_framework(model)

    # Example data (replace with real data in practice)
    data = jax.random.normal(key, (100, 5))
    labels = jax.random.randint(key, (100,), 0, 2)

    # Detect bias
    bias_metrics = ethical_framework.detect_bias(data, sensitive_attributes=[0])
    print("Bias metrics:", bias_metrics)

    # Add a simple fairness constraint (example)
    def demographic_parity(params, x, y):
        preds = nn.softmax(nn.Dense(10).apply({'params': params}, x))
        return jnp.abs(preds[y == 0].mean() - preds[y == 1].mean())

    ethical_framework.add_fairness_constraint(demographic_parity, weight=0.1)

    # Train with fairness
    ethical_framework.train_with_fairness(data, labels)

    # Generate transparency report
    report = ethical_framework.generate_transparency_report(data, labels)
    print("Transparency report:", report)
