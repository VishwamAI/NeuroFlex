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

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from typing import List, Tuple, Optional
from tqdm import tqdm
from NeuroFlex.utils.utils import normalize_data, preprocess_data

class CNN(nn.Module):
    """
    A Convolutional Neural Network (CNN) class for NeuroFlex.

    This module encapsulates the CNN-specific functionality with self-healing
    mechanisms and adaptive learning rate adjustments.

    Args:
        features (List[int]): The number of filters in each convolutional layer.
        conv_dim (int): The dimension of convolution (2 or 3).
        input_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Initial learning rate for optimization.
    """
    def __init__(self, features: List[int], conv_dim: int, input_channels: int, num_classes: int,
                 dropout_rate: float = 0.5, learning_rate: float = 0.001):
        super().__init__()
        self.features = features
        self.conv_dim = conv_dim
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
        in_channels = input_channels
        for feat in features:
            self.layers.append(nn.Conv2d(in_channels, feat, kernel_size=3, padding=1) if conv_dim == 2
                               else nn.Conv3d(in_channels, feat, kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.MaxPool2d(2) if conv_dim == 2 else nn.MaxPool3d(2))
            self.layers.append(nn.Dropout(self.dropout_rate))
            in_channels = feat

        self.fc = nn.Linear(features[-1], num_classes)
        self.to(self.device)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

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
                self.train_model()
            elif issue == "Model performance is below threshold":
                self.improve_model()
            elif issue == "Model hasn't been updated in 24 hours":
                self.update_model()
            elif issue == "Gradient explosion detected":
                self.handle_gradient_explosion()
            elif issue == "Model is stuck in local minimum":
                self.escape_local_minimum()

    def train_model(self, X, y, epochs=100, batch_size=32, patience=10):
        print("Training model...")
        self.is_trained = True
        self.last_update = time.time()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        best_loss = float('inf')
        no_improve = 0

        for epoch in tqdm(range(epochs), desc="Training"):
            self.train()
            epoch_loss = 0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(dataloader)
            if avg_loss < best_loss:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.update_performance()

    def improve_model(self):
        print("Improving model performance...")
        self.performance = min(self.performance * 1.2 + 0.01, 1.0)
        self.update_performance()

    def update_model(self):
        print("Updating model...")
        self.last_update = time.time()
        self.update_performance()

    def handle_gradient_explosion(self):
        print("Handling gradient explosion...")
        self.learning_rate *= 0.5

    def escape_local_minimum(self):
        print("Attempting to escape local minimum...")
        self.learning_rate = min(self.learning_rate * 2.5, 0.1)
        print(f"New learning rate: {self.learning_rate}")

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

def create_cnn(features: List[int], conv_dim: int, input_channels: int, num_classes: int,
               dropout_rate: float = 0.5, learning_rate: float = 0.001) -> CNN:
    """
    Factory function to create a CNN instance.

    Args:
        features (List[int]): The number of filters in each convolutional layer.
        conv_dim (int): The dimension of convolution (2 or 3).
        input_channels (int): The number of input channels.
        num_classes (int): The number of output classes.
        dropout_rate (float): Dropout rate for regularization.
        learning_rate (float): Initial learning rate for optimization.

    Returns:
        CNN: An instance of the CNN.
    """
    return CNN(features, conv_dim, input_channels, num_classes, dropout_rate, learning_rate)

def calculate_accuracy(model, X, y):
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(model.device)
        y_tensor = torch.LongTensor(y).to(model.device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_tensor).sum().item()
        total = y_tensor.size(0)
        accuracy = correct / total
    return accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def load_model(model, path):
    model.load_state_dict(torch.load(path))
    return model
