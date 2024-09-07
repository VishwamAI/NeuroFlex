import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Optional, Callable
import logging
import time

logging.basicConfig(level=logging.INFO)

class PyTorchModel(nn.Module):
    def __init__(self, input_dim: int, hidden_layers: List[int], output_dim: int,
                 dropout_rate: float = 0.5, learning_rate: float = 0.001):
        super(PyTorchModel, self).__init__()
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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.layers = nn.ModuleList()
        in_features = input_dim
        for units in hidden_layers:
            self.layers.append(nn.Linear(in_features, units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(self.dropout_rate))
            in_features = units
        self.layers.append(nn.Linear(in_features, output_dim))
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

def create_pytorch_model(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]) -> PyTorchModel:
    return PyTorchModel(input_shape[0], hidden_layers, output_dim)

def train_pytorch_model(model: PyTorchModel,
                        x_train: torch.Tensor,
                        y_train: torch.Tensor,
                        epochs: int = 10,
                        batch_size: int = 32,
                        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        callback: Optional[Callable[[float], None]] = None) -> dict:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

    history = {'train_loss': [], 'val_loss': []}

    num_samples = x_train.shape[0]
    num_batches = max(1, num_samples // batch_size)

    logging.info(f"PyTorch model initial parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_x = x_train[i:i+batch_size].to(model.device)
            batch_y = y_train[i:i+batch_size].to(model.device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        logging.info(f"PyTorch - Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        if callback:
            callback(avg_loss)

        model.gradient_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        model.performance = 1.0 - avg_loss  # Simple performance metric
        model.update_performance()
        model.adjust_learning_rate()

        issues = model.diagnose()
        if issues:
            logging.info(f"Diagnosed issues: {issues}")
            model.heal(issues)

        # Compute epoch loss
        model.eval()
        with torch.no_grad():
            train_loss = criterion(model(x_train), y_train).item()
            history['train_loss'].append(train_loss)

            if validation_data:
                val_x, val_y = validation_data
                val_loss = criterion(model(val_x), val_y).item()
                history['val_loss'].append(val_loss)

    model.is_trained = True
    model.last_update = time.time()
    logging.info(f"PyTorch model final parameters: {sum(p.numel() for p in model.parameters())}")

    return history

def pytorch_predict(model: PyTorchModel, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x.to(model.device))
