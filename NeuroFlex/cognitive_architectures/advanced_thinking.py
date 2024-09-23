import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from typing import Dict, Tuple


class CDSTDP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        learning_rate: float = 0.001,
    ):
        super(CDSTDP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Initialize synaptic weights
        self.synaptic_weights = nn.Parameter(torch.randn(hidden_size, hidden_size))

        # Initialize optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        # Performance tracking
        self.performance = 0.0
        self.last_update = 0
        self.performance_history = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        x = torch.relu(self.hidden_layer(x))
        x = self.output_layer(x)
        return x

    def update_synaptic_weights(
        self, pre_synaptic: torch.Tensor, post_synaptic: torch.Tensor, dopamine: float
    ):
        # Ensure pre_synaptic and post_synaptic have the same batch size
        assert pre_synaptic.size(0) == post_synaptic.size(0), "Batch sizes must match"

        # Implement STDP rule
        time_window = 21  # -10 to 10, inclusive
        delta_t = (
            torch.arange(-10, 11)
            .unsqueeze(0)
            .unsqueeze(1)
            .unsqueeze(2)
            .repeat(
                pre_synaptic.size(0), pre_synaptic.size(1), post_synaptic.size(1), 1
            )
        )
        stdp = torch.where(
            delta_t > 0,
            torch.exp(-delta_t / 20.0) * 0.1,
            torch.exp(delta_t / 20.0) * -0.12,
        )

        # Modulate STDP by dopamine
        modulated_stdp = stdp * dopamine

        # Compute weight updates
        pre_expanded = (
            pre_synaptic.unsqueeze(2)
            .unsqueeze(3)
            .expand(-1, -1, post_synaptic.size(1), time_window)
        )
        post_expanded = (
            post_synaptic.unsqueeze(1)
            .unsqueeze(3)
            .expand(-1, pre_synaptic.size(1), -1, time_window)
        )
        dw = (pre_expanded * post_expanded * modulated_stdp).sum(dim=3)

        # Update synaptic weights
        self.synaptic_weights.data += dw.mean(dim=0)  # Average over batch

    def train_step(self, inputs: torch.Tensor, targets: torch.Tensor, dopamine: float):
        self.optimizer.zero_grad()
        outputs = self(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()

        # Apply CDSTDP
        with torch.no_grad():
            pre_synaptic = self.input_layer(inputs)
            post_synaptic = self.hidden_layer(pre_synaptic)
            self.update_synaptic_weights(pre_synaptic, post_synaptic, dopamine)

        self.optimizer.step()
        return loss.item()

    def diagnose(self) -> Dict[str, bool]:
        current_time = time.time()
        issues = {
            "low_performance": self.performance < 0.8,
            "stagnant_performance": len(self.performance_history) > 10
            and np.mean(self.performance_history[-10:])
            < np.mean(self.performance_history[-20:-10]),
            "needs_update": (
                current_time - self.last_update > 86400
            ),  # 24 hours in seconds
        }
        return issues

    def heal(self, inputs: torch.Tensor, targets: torch.Tensor):
        issues = self.diagnose()
        if issues["low_performance"] or issues["stagnant_performance"]:
            # Increase learning rate temporarily
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate * 2

            # Perform additional training
            for _ in range(100):
                loss = self.train_step(inputs, targets, dopamine=1.0)

            # Reset learning rate
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        if issues["needs_update"]:
            self.last_update = time.time()

        self.performance = self.evaluate(inputs, targets)
        self.performance_history.append(self.performance)

    def evaluate(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            outputs = self(inputs)
            loss = nn.functional.mse_loss(outputs, targets)
        performance = 1.0 / (
            1.0 + loss.item()
        )  # Convert loss to performance metric (0 to 1)
        return performance


def create_cdstdp(
    input_size: int, hidden_size: int, output_size: int, learning_rate: float = 0.001
) -> CDSTDP:
    return CDSTDP(input_size, hidden_size, output_size, learning_rate)
