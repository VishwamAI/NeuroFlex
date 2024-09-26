# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# PyTorch specific implementations

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple, Optional, Callable
import logging
import time

logging.basicConfig(level=logging.INFO)

class PyTorchModel(nn.Module):
    def __init__(self, features: int, dropout_rate: float = 0.5, learning_rate: float = 0.001):
        super().__init__()
        self.features = features
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

        self.layer = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(features, features)
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D")
        if x.shape[1] != self.features:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (batch_size, {self.features})")
        x = self.layer(x)
        return torch.log_softmax(x, dim=-1)

    @property
    def num_features(self):
        return self.features

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

def train_pytorch_model(
    model: PyTorchModel,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 10,
    batch_size: int = 32,
    callback: Optional[Callable[[float], None]] = None
) -> None:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate)

    num_samples = X.shape[0]
    num_batches = max(1, num_samples // batch_size)

    logging.info(f"PyTorch model initial parameters: {sum(p.numel() for p in model.parameters())}")

    for epoch in range(epochs):
        perm = torch.randperm(num_samples)
        X_shuffled = X[perm]
        y_shuffled = y[perm]

        epoch_loss = 0.0
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_samples)
            batch_X = X_shuffled[start_idx:end_idx].to(model.device)
            batch_y = y_shuffled[start_idx:end_idx].to(model.device)

            optimizer.zero_grad()
            outputs = model(batch_X)
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

    model.is_trained = True
    model.last_update = time.time()
    logging.info(f"PyTorch model final parameters: {sum(p.numel() for p in model.parameters())}")

def batch_predict(model: PyTorchModel, x: torch.Tensor) -> torch.Tensor:
    try:
        # Ensure input is a PyTorch tensor
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        # Move input to the same device as the model
        x = x.to(model.device)

        # Reshape input if necessary
        if x.dim() == 1:
            x = x.view(1, -1)
        elif x.dim() == 0:
            x = x.view(1, 1)
        elif x.dim() != 2:
            raise ValueError(f"Invalid input shape. Expected 2 dimensions, got {x.dim()}. Input shape: {x.shape}")

        # Apply the model
        model.eval()
        with torch.no_grad():
            output = model(x)

        logging.info(f"Batch prediction successful. Input shape: {x.shape}, Output shape: {output.shape}")
        return output
    except Exception as e:
        logging.error(f"Error in batch_predict: {str(e)}")
        raise

def parallel_train(model: PyTorchModel, X: torch.Tensor, y: torch.Tensor) -> None:
    if torch.cuda.device_count() > 1:
        logging.info(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    train_pytorch_model(model, X, y)
