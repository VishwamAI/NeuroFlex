import torch
import torch.nn as nn
import numpy as np
import jax
import jax.numpy as jnp
import optax
from flax import linen as flax_nn

class PyTorchModel(nn.Module):
    def __init__(self, features):
        super(PyTorchModel, self).__init__()
        self.layer = nn.Linear(features, features)

    def forward(self, x):
        x = self.layer(x)
        x = torch.relu(x)
        return torch.log_softmax(x, dim=-1)

def train_pytorch_model(model, X, y, epochs=10, learning_rate=0.01, batch_size=32, callback=None):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    num_samples = X.shape[0]
    num_batches = max(1, num_samples // batch_size)

    for epoch in range(epochs):
        # Shuffle the data
        perm = torch.randperm(num_samples)
        X_shuffled = X_tensor[perm]
        y_shuffled = y_tensor[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_X = X_shuffled[start_idx:end_idx]
            batch_y = y_shuffled[start_idx:end_idx]

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"PyTorch - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if callback:
            callback(avg_loss, model)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"PyTorch model parameters at epoch {epoch+1}:")
            for name, param in model.named_parameters():
                print(f"{name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")

    return model.state_dict()

def jax_to_torch_params(jax_params):
    torch_params = {}
    for key, value in jax_params.items():
        if isinstance(value, dict):
            torch_params[key] = jax_to_torch_params(value)
        else:
            torch_params[key] = torch.FloatTensor(np.array(value))
    return torch_params

def torch_to_jax_params(torch_params):
    jax_params = {}
    for key, value in torch_params.items():
        if isinstance(value, dict):
            jax_params[key] = torch_to_jax_params(value)
        else:
            jax_params[key] = jnp.array(value.detach().numpy())
    return jax_params
