import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional

class PyTorchModel(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]):
        super(PyTorchModel, self).__init__()
        self.input_dim = input_shape[0]  # Assuming 1D input
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(nn.Linear(self.input_dim, hidden_layers[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            self.layers.append(nn.ReLU())

        # Output layer
        self.layers.append(nn.Linear(hidden_layers[-1], output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def create_pytorch_model(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]) -> PyTorchModel:
    return PyTorchModel(input_shape, output_dim, hidden_layers)

def train_pytorch_model(model: PyTorchModel,
                        x_train: torch.Tensor,
                        y_train: torch.Tensor,
                        epochs: int = 10,
                        batch_size: int = 32,
                        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> dict:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        model.train()
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            history['train_loss'].append(train_loss)
            if validation_data:
                val_x, val_y = validation_data
                val_loss = criterion(model(val_x), val_y).item()
                history['val_loss'].append(val_loss)

    return history

def pytorch_predict(model: PyTorchModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)
