import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Optional
from flax.training import train_state
import optax

class JaxModel(nn.Module):
    hidden_layers: List[int]
    output_dim: int

    @nn.compact
    def __call__(self, x):
        for units in self.hidden_layers:
            x = nn.Dense(units)(x)
            x = nn.relu(x)
        x = nn.Dense(self.output_dim)(x)
        return x

def create_jax_model(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]) -> JaxModel:
    return JaxModel(hidden_layers=hidden_layers, output_dim=output_dim)

def train_jax_model(model: JaxModel,
                    x_train: jnp.ndarray,
                    y_train: jnp.ndarray,
                    input_shape: Tuple[int, ...],
                    epochs: int = 10,
                    batch_size: int = 32,
                    learning_rate: float = 1e-3,
                    validation_data: Optional[Tuple[jnp.ndarray, jnp.ndarray]] = None):

    rng = jax.random.PRNGKey(0)

    @jax.jit
    def loss_fn(params, x, y):
        pred = model.apply({'params': params}, x)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def train_step(state, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(state.params, x, y)
        state = state.apply_gradients(grads=grads)
        return state, loss

    params = model.init(rng, jnp.ones((1,) + input_shape))['params']
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            state, loss = train_step(state, batch_x, batch_y)

        train_loss = loss_fn(state.params, x_train, y_train)
        history['train_loss'].append(train_loss)

        if validation_data:
            val_x, val_y = validation_data
            val_loss = loss_fn(state.params, val_x, val_y)
            history['val_loss'].append(val_loss)

    return state, history

@jax.jit
def jax_predict(model: JaxModel, params, x: jnp.ndarray) -> jnp.ndarray:
    return model.apply({'params': params}, x)
