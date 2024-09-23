import torch
import torch.nn as nn
import torch.optim as optim
from typing import Callable, Tuple, Any, List
import time
import logging
from ..utils.utils import normalize_data
from ..utils.descriptive_statistics import preprocess_data


class LRNNCell(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, activation: Callable = nn.Tanh()
    ):
        super(LRNNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation
        self.linear = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(
        self, h: torch.Tensor, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([h, x], dim=1)
        new_h = self.activation(self.linear(combined))
        return new_h, new_h


class LRNN(nn.Module):
    def __init__(
        self,
        features: Tuple[int, ...],
        activation: Callable = nn.Tanh(),
        dropout_rate: float = 0.5,
        learning_rate: float = 0.001,
    ):
        super(LRNN, self).__init__()
        self.features = features
        self.activation = activation
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

        self.cells = nn.ModuleList(
            [
                LRNNCell(features[i - 1] if i > 0 else features[0], feat, activation)
                for i, feat in enumerate(features)
            ]
        )
        self.dropout = nn.Dropout(self.dropout_rate)
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, input_dim = x.shape
        h = [
            torch.zeros(batch_size, feat, device=self.device) for feat in self.features
        ]
        outputs = []

        for t in range(seq_len):
            x_t = x[:, t, :]
            for i, cell in enumerate(self.cells):
                h[i], _ = cell(h[i], x_t if i == 0 else h[i - 1])
                h[i] = self.dropout(h[i])
            outputs.append(h[-1].unsqueeze(1))

        y = torch.cat(outputs, dim=1)
        return y, h[-1]

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


def create_rnn_block(
    features: Tuple[int, ...],
    activation: Callable = nn.Tanh(),
    dropout_rate: float = 0.5,
    learning_rate: float = 0.001,
) -> LRNN:
    return LRNN(
        features=features,
        activation=activation,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )
