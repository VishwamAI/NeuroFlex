import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class PyTorchModel(nn.Module):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]):
        super(PyTorchModel, self).__init__()
        self.input_layer = nn.Flatten()
        self.hidden_layers = nn.ModuleList([
            nn.Linear(input_shape[0] if i == 0 else hidden_layers[i-1], units)
            for i, units in enumerate(hidden_layers)
        ])
        self.output_layer = nn.Linear(hidden_layers[-1], output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))
        return self.output_layer(x)

def create_pytorch_model(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]) -> PyTorchModel:
    return PyTorchModel(input_shape, output_dim, hidden_layers)

def train_pytorch_model(model: PyTorchModel,
                        x_train: torch.Tensor,
                        y_train: torch.Tensor,
                        epochs: int = 10,
                        batch_size: int = 32,
                        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> dict:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

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

        # Compute epoch loss
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
