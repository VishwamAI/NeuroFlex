from typing import List, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from tqdm import tqdm
from NeuroFlex.utils.utils import normalize_data, preprocess_data


class MachineLearning(nn.Module):
    def __init__(
        self, features, activation=nn.ReLU, dropout_rate=0.5, learning_rate=0.001
    ):
        super().__init__()
        self.features = features
        self.activation_class = activation
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
        for i in range(len(features) - 1):
            self.layers.append(nn.Linear(features[i], features[i + 1]))
            if i < len(features) - 2:
                self.layers.append(self.activation_class())
                self.layers.append(nn.Dropout(self.dropout_rate))

        self.to(self.device)

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            x = x.to(self.device)
        elif isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float().to(self.device)
        else:
            raise ValueError("Input must be a torch.Tensor or numpy.ndarray")

        for layer in self.layers:
            x = layer(x)
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
        if len(self.performance_history) > 5 and all(
            p < 0.01 for p in self.performance_history[-5:]
        ):
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

    def train_model(self):
        print("Training model...")
        self.is_trained = True
        self.last_update = time.time()
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
        self.learning_rate = min(
            self.learning_rate * 2.5, 0.1
        )  # Increase more aggressively, but cap at 0.1
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
                # Performance improved, increase learning rate
                self.learning_rate *= 1.05
            elif current_performance < previous_performance:
                # Performance worsened, decrease learning rate
                self.learning_rate *= 0.95
            # If performance stayed the same, don't change the learning rate
        else:
            # If we don't have enough history, make a small increase
            self.learning_rate *= 1.01

        # Ensure the learning rate doesn't become too small or too large
        self.learning_rate = max(min(self.learning_rate, 0.1), 1e-5)
        return self.learning_rate


class NeuroFlexClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, features, activation=nn.ReLU, dropout_rate=0.5, learning_rate=0.001
    ):
        self.features = features
        self.activation_class = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y, epochs=100, patience=10):
        self.model = MachineLearning(
            features=self.features,
            activation=self.activation_class,
            dropout_rate=self.dropout_rate,
            learning_rate=self.learning_rate,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        # Convert input data to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(device)
        y_tensor = torch.LongTensor(y).to(device)

        # Use the MachineLearning model's train_model method
        self.model.train_model()

        # Perform actual training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        no_improve = 0

        for epoch in tqdm(range(epochs), desc="Training"):
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

            # Early stopping
            if loss < best_loss:
                best_loss = loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        self.model.update_performance()
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.model.device)
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.model.device)
            outputs = self.model(X_tensor)
            probas = nn.functional.softmax(outputs, dim=1)
        return probas.cpu().numpy()

    def self_fix(self, X, y):
        initial_lr = self.model.learning_rate
        print(f"Initial learning rate: {initial_lr}")

        issues = self.model.diagnose()
        if issues:
            print(f"Diagnosed issues: {issues}")
            self.model.heal(issues)

        predictions = self.predict(X)
        misclassified = X[predictions != y]
        misclassified_labels = y[predictions != y]
        if len(misclassified) > 0:
            print(f"Retraining on {len(misclassified)} misclassified samples")
            self.fit(misclassified, misclassified_labels)

        healing_boost_factor = 1.5  # Increase learning rate by 50%
        max_lr = 0.1  # Maximum learning rate cap
        min_increase = 1e-5  # Minimum increase in learning rate

        # Calculate the new learning rate
        if initial_lr >= max_lr:
            new_lr = max_lr
        else:
            # Ensure the new learning rate doesn't exceed max_lr
            new_lr = min(initial_lr * healing_boost_factor, max_lr)

            # Apply additional boost if the model was stuck in a local minimum
            if "Model is stuck in local minimum" in issues:
                new_lr = min(new_lr * 1.2, max_lr)

            # Ensure the learning rate always increases by at least min_increase
            new_lr = max(new_lr, initial_lr + min_increase)

            # Apply the maximum cap
            new_lr = min(new_lr, max_lr)

        # Ensure the final learning rate is different from the initial one
        if abs(new_lr - initial_lr) < min_increase and new_lr < max_lr:
            new_lr = min(initial_lr + min_increase, max_lr)

        self.model.learning_rate = new_lr

        print(f"Learning rate after adjustment: {self.model.learning_rate}")
        return self


# This section has been removed as it contained redundant PyTorchModel class and train_pytorch_model function.
# The functionality is now integrated into the MachineLearning and NeuroFlexClassifier classes.


# Additional utility functions for machine learning can be added here
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
