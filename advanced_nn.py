import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import flax.linen as nn
from flax.training import train_state
import optax
import numpy as np
from typing import Sequence, Callable
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

class AdvancedNN(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu
    dropout_rate: float = 0.5
    fairness_constraint: float = 0.1  # New parameter for fairness constraint

    @nn.compact
    def __call__(self, x, training: bool = False, sensitive_attribute: jnp.ndarray = None):
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        # Apply fairness constraint
        if sensitive_attribute is not None:
            x = self.apply_fairness_constraint(x, sensitive_attribute)

        x = nn.Dense(self.features[-1])(x)
        return x

    def feature_importance(self, x):
        activations = []
        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
            activations.append(x)
        return activations

    def apply_fairness_constraint(self, x, sensitive_attribute, fairness_constraint):
        # Implement a simple demographic parity constraint
        group_means = jnp.mean(x, axis=0, keepdims=True)
        overall_mean = jnp.mean(group_means, axis=1, keepdims=True)
        adjusted_x = x + fairness_constraint * (overall_mean - group_means[sensitive_attribute])
        return adjusted_x

def create_train_state(rng, model, input_shape, learning_rate):
    params = model.init(rng, jnp.ones(input_shape))['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jit
def train_step(state, batch):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch['image'])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits, batch['label']).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params}, batch['image'])
    return jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])

# Interpretability features using SHAP
def interpret_model(model, params, input_data):
    import shap

    # Create a function that the explainer can call
    def model_predict(x):
        return model.apply({'params': params}, x)

    # Create the explainer
    explainer = shap.KernelExplainer(model_predict, input_data)

    # Calculate SHAP values
    shap_values = explainer.shap_values(input_data)

    # Visualize the results
    shap.summary_plot(shap_values, input_data)

    return shap_values

# Implement FGSM for adversarial training
def adversarial_training(model, params, input_data, epsilon):
    def loss_fn(params, x, y):
        logits = model.apply({'params': params}, x)
        return optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()

    grad_fn = jax.grad(loss_fn, argnums=1)

    def fgsm_attack(params, x, y):
        grads = grad_fn(params, x, y)
        return x + epsilon * jnp.sign(grads)

    perturbed_input = fgsm_attack(params, input_data['image'], input_data['label'])
    return {'image': perturbed_input, 'label': input_data['label']}

# Main training loop with generalization techniques and fairness considerations
def train_model(model, train_data, val_data, num_epochs, batch_size, learning_rate, fairness_constraint, patience=5, epsilon=0.1):
    rng = jax.random.PRNGKey(0)
    state = create_train_state(rng, model, train_data['image'].shape[1:], learning_rate)

    best_val_accuracy = 0
    patience_counter = 0

    # Initialize fairness metrics
    fairness_metric = BinaryLabelDatasetMetric(
        train_data, label_name='label', protected_attribute_names=['sensitive_attr']
    )
    initial_disparate_impact = fairness_metric.disparate_impact()

    for epoch in range(num_epochs):
        # Training
        for batch in get_batches(train_data, batch_size):
            # Data augmentation
            batch['image'] = data_augmentation(batch['image'])

            # Adversarial training
            adv_batch = adversarial_training(model, state.params, batch, epsilon)

            # Apply bias mitigation
            reweighing = Reweighing(unprivileged_groups=[{'sensitive_attr': 0}],
                                    privileged_groups=[{'sensitive_attr': 1}])
            mitigated_batch = reweighing.fit_transform(BinaryLabelDataset(
                df=adv_batch,
                label_names=['label'],
                protected_attribute_names=['sensitive_attr']
            ))

            # Apply fairness constraint
            mitigated_batch['image'] = model.apply_fairness_constraint(
                mitigated_batch['image'],
                mitigated_batch['sensitive_attr']
            )

            state, loss = train_step(state, mitigated_batch)

        # Validation
        val_accuracy = 0
        for batch in get_batches(val_data, batch_size):
            val_accuracy += eval_step(state, batch)
        val_accuracy /= len(val_data['image']) // batch_size

        # Compute fairness metrics
        fairness_metric = BinaryLabelDatasetMetric(
            val_data, label_name='label', protected_attribute_names=['sensitive_attr']
        )
        current_disparate_impact = fairness_metric.disparate_impact()

        print(f"Epoch {epoch}: loss = {loss:.3f}, val_accuracy = {val_accuracy:.3f}, "
              f"disparate_impact = {current_disparate_impact:.3f}")

        # Early stopping (considering both accuracy and fairness)
        if val_accuracy > best_val_accuracy and current_disparate_impact > initial_disparate_impact:
            best_val_accuracy = val_accuracy
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return state

def data_augmentation(images):
    # Implement data augmentation techniques here
    # For example, random flips, rotations, etc.
    return images  # Placeholder, replace with actual augmentation

# Utility function to get batches
def get_batches(data, batch_size):
    for i in range(0, len(data['image']), batch_size):
        yield {k: v[i:i+batch_size] for k, v in data.items()}

# Example usage
if __name__ == "__main__":
    # Define model architecture
    model = AdvancedNN([64, 32, 10])

    # Generate dummy data
    num_samples = 1000
    input_dim = 784  # e.g., for MNIST
    num_classes = 10
    rng = np.random.default_rng(0)

    # Generate dummy data with sensitive attributes
    train_data = {
        'image': rng.normal(size=(num_samples, input_dim)).astype(np.float32),
        'label': rng.integers(0, num_classes, size=(num_samples,)),
        'sensitive_attr': rng.integers(0, 2, size=(num_samples,))  # Binary sensitive attribute
    }
    val_data = {
        'image': rng.normal(size=(num_samples // 5, input_dim)).astype(np.float32),
        'label': rng.integers(0, num_classes, size=(num_samples // 5,)),
        'sensitive_attr': rng.integers(0, 2, size=(num_samples // 5,))
    }

    # Train the model with fairness constraints
    trained_state = train_model(model, train_data, val_data, num_epochs=10, batch_size=32, learning_rate=1e-3,
                                fairness_constraint=0.1, epsilon=0.1)

    print("Training completed.")

    # Evaluate model fairness
    def evaluate_fairness(state, data):
        # Predict labels using the trained model
        logits = state.apply_fn({'params': state.params}, data['image'])
        predicted_labels = jnp.argmax(logits, axis=-1)

        # Create a BinaryLabelDataset
        dataset = BinaryLabelDataset(
            df=data,
            label_names=['label'],
            protected_attribute_names=['sensitive_attr'],
            favorable_label=1,
            unfavorable_label=0
        )

        # Create a new dataset with predicted labels
        predicted_dataset = dataset.copy()
        predicted_dataset.labels = predicted_labels

        # Compute fairness metrics
        metric = BinaryLabelDatasetMetric(predicted_dataset,
                                          unprivileged_groups=[{'sensitive_attr': 0}],
                                          privileged_groups=[{'sensitive_attr': 1}])

        return {
            'disparate_impact': metric.disparate_impact(),
            'equal_opportunity_difference': metric.equal_opportunity_difference()
        }

    fairness_metrics = evaluate_fairness(trained_state, val_data)
    print("Fairness metrics:", fairness_metrics)

    # Interpret model decisions
    shap_values = interpret_model(model, trained_state.params, val_data['image'][:100])
    print("SHAP values computed for model interpretation.")
