import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax

class AdaptiveLearningSystem:
    def __init__(self, model, learning_rate=0.001, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer = optax.sgd(learning_rate=self.learning_rate, momentum=self.momentum)
        self.state = None

    def init(self, rng, input_shape):
        variables = self.model.init(rng, jnp.ones(input_shape))
        self.state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables['params'],
            tx=self.optimizer
        )

    def adapt_learning_rate(self, device_constraints):
        # Adjust learning rate based on device constraints
        cpu_usage = device_constraints.get('cpu_usage', 0.5)
        memory_usage = device_constraints.get('memory_usage', 0.5)
        battery_level = device_constraints.get('battery_level', 1.0)

        # Simple adaptive strategy: reduce learning rate when resources are constrained
        adaptation_factor = (1 - cpu_usage) * (1 - memory_usage) * battery_level
        new_learning_rate = self.learning_rate * adaptation_factor

        # Update optimizer with new learning rate
        self.optimizer = optax.sgd(learning_rate=new_learning_rate, momentum=self.momentum)
        self.state = self.state.replace(tx=self.optimizer)

    @jax.jit
    def train_step(self, state, batch):
        def loss_fn(params):
            logits = self.model.apply({'params': params}, batch['image'])
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, batch['label']).mean()
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train_epoch(self, train_ds, batch_size, device_constraints):
        train_ds_size = len(train_ds['image'])
        steps_per_epoch = train_ds_size // batch_size

        self.adapt_learning_rate(device_constraints)

        for _ in range(steps_per_epoch):
            batch = {k: v[batch_size:] for k, v in train_ds.items()}
            self.state, loss = self.train_step(self.state, batch)

        return loss

    @jax.jit
    def evaluate_step(self, params, batch):
        logits = self.model.apply({'params': params}, batch['image'])
        return jnp.mean(jnp.argmax(logits, axis=-1) == batch['label'])

    def evaluate_model(self, test_ds):
        accuracy = jax.jit(lambda params, batch: self.evaluate_step(params, batch))(self.state.params, test_ds)
        return accuracy

def create_cnn_model():
    return nn.Sequential([
        nn.Conv(features=32, kernel_size=(3, 3)),
        nn.relu,
        nn.avg_pool,
        nn.Conv(features=64, kernel_size=(3, 3)),
        nn.relu,
        nn.avg_pool,
        nn.Flatten(),
        nn.Dense(features=256),
        nn.relu,
        nn.Dense(features=10),
    ])

# Example usage:
# rng = jax.random.PRNGKey(0)
# model = create_cnn_model()
# adaptive_system = AdaptiveLearningSystem(model)
# adaptive_system.init(rng, (1, 28, 28, 1))  # For MNIST-like dataset
#
# # Training loop
# for epoch in range(num_epochs):
#     device_constraints = get_device_constraints()  # Function to get real-time device constraints
#     loss = adaptive_system.train_epoch(train_ds, batch_size, device_constraints)
#     accuracy = adaptive_system.evaluate_model(test_ds)
#     print(f"Epoch {epoch}: Loss = {loss}, Accuracy = {accuracy}")
