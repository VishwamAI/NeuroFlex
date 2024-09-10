import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Any, Union, Tuple
import time
import logging
from tqdm import tqdm
from ..constants import PERFORMANCE_THRESHOLD, UPDATE_INTERVAL, LEARNING_RATE_ADJUSTMENT, MAX_HEALING_ATTEMPTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MultiModalLearning(nn.Module):
    def __init__(self, output_dim: int = 10):
        super().__init__()
        self.modalities = {}
        self.fusion_method = None
        self.performance = 0.0
        self.last_update = time.time()
        self.performance_history = []
        self.learning_rate = 0.001
        self.performance_threshold = PERFORMANCE_THRESHOLD
        self.update_interval = UPDATE_INTERVAL
        self.max_healing_attempts = MAX_HEALING_ATTEMPTS
        self.classifier = nn.Linear(128, output_dim)  # Assuming fused dimension is 128
        self.train_loss_history = []
        self.val_loss_history = []
        self.healing_strategies = [
            self._adjust_learning_rate,
            self._reinitialize_layers,
            self._increase_model_capacity,
            self._apply_regularization
        ]

    def add_modality(self, name: str, input_shape: tuple):
        """Add a new modality to the multi-modal learning model."""
        self.modalities[name] = {
            'input_shape': input_shape,
            'encoder': self._create_encoder(input_shape)
        }

    def _create_encoder(self, input_shape: tuple) -> nn.Module:
        """Create a simple encoder network for a given input shape."""
        return nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )

    def set_fusion_method(self, method: str):
        """Set the fusion method for combining modalities."""
        supported_methods = ['concatenation', 'attention']
        if method not in supported_methods:
            raise ValueError(f"Unsupported fusion method. Choose from {supported_methods}")
        self.fusion_method = method

    def fuse_modalities(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse the encoded modalities based on the selected fusion method."""
        if self.fusion_method == 'concatenation':
            return torch.cat(list(encoded_modalities.values()), dim=1)
        elif self.fusion_method == 'attention':
            # Implement attention-based fusion
            # This is a placeholder and should be implemented based on specific requirements
            return torch.stack(list(encoded_modalities.values())).mean(dim=0)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the multi-modal learning model."""
        if set(inputs.keys()) != set(self.modalities.keys()):
            raise ValueError("Input modalities do not match the defined modalities")

        encoded_modalities = {}
        for name, modality in self.modalities.items():
            encoded_modalities[name] = modality['encoder'](inputs[name].view(inputs[name].size(0), -1))

        fused = self.fuse_modalities(encoded_modalities)
        # Remove the classifier layer to keep the output dimension at 128
        return fused

    def fit(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, val_data: Dict[str, torch.Tensor] = None, val_labels: torch.Tensor = None, epochs: int = 10, lr: float = 0.001, patience: int = 5, batch_size: int = 32):
        """Train the multi-modal learning model."""
        if val_data is None or val_labels is None:
            # Split data into training and validation sets if validation data is not provided
            train_data, val_data, train_labels, val_labels = self._split_data(data, labels)
        else:
            train_data, train_labels = data, labels

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Starting training with initial learning rate: {lr}")

        moving_avg_window = min(10, epochs)
        performance_window = []
        best_performance = 0
        no_improve_count = 0
        best_epoch = 0

        for epoch in range(epochs):
            self.train()  # Set model to training mode
            train_loss, train_accuracy = self._train_epoch(train_data, train_labels, optimizer, criterion, batch_size)
            self.train_loss_history.append(train_loss)

            val_loss, val_accuracy = self._validate(val_data, val_labels, criterion, batch_size)
            self.val_loss_history.append(val_loss)

            current_performance = val_accuracy
            performance_window.append(current_performance)
            if len(performance_window) > moving_avg_window:
                performance_window.pop(0)

            moving_avg_performance = sum(performance_window) / len(performance_window)
            previous_performance = self.performance
            self._update_performance(moving_avg_performance)

            logger.info(f"Epoch {epoch + 1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
            logger.info(f"  Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.6f}")
            logger.info(f"  Current Performance (Moving Avg): {moving_avg_performance:.6f}")
            logger.info(f"  Updated Overall Performance: {self.performance:.6f}")
            logger.info(f"  Performance Change: {self.performance - previous_performance:.6f}")

            # Log model parameters and gradients
            for name, param in self.named_parameters():
                if param.requires_grad:
                    grad_norm = param.grad.norm().item() if param.grad is not None else 0
                    logger.debug(f"  {name} - Grad: {grad_norm:.6f}, Value: {param.data.norm().item():.6f}")

            scheduler.step(moving_avg_performance)

            if moving_avg_performance > best_performance:
                best_performance = moving_avg_performance
                best_epoch = epoch
                no_improve_count = 0
                self._save_best_model()
                logger.info("New best model saved.")
            else:
                no_improve_count += 1

            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            if moving_avg_performance < self.performance_threshold:
                logger.warning(f"Moving average performance {moving_avg_performance:.4f} below threshold {self.performance_threshold}. Initiating self-healing.")
                self._self_heal()
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
                logger.info(f"After self-healing - Learning Rate: {self.learning_rate:.6f}, Performance: {self.performance:.4f}")

        self._load_best_model()
        logger.info(f"Training completed. Best Performance: {best_performance:.4f} at epoch {best_epoch + 1}")
        logger.info(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Performance History: {self.performance_history}")
        self._plot_training_history()

    def _split_data(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, val_ratio: float = 0.2) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Split data into training and validation sets."""
        num_samples = labels.size(0)
        num_val_samples = int(num_samples * val_ratio)
        indices = torch.randperm(num_samples)
        val_indices = indices[:num_val_samples]
        train_indices = indices[num_val_samples:]

        train_data = {k: v[train_indices] for k, v in data.items()}
        val_data = {k: v[val_indices] for k, v in data.items()}
        train_labels = labels[train_indices]
        val_labels = labels[val_indices]

        return train_data, val_data, train_labels, val_labels

    def _train_epoch(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, optimizer: torch.optim.Optimizer, criterion: nn.Module, batch_size: int = 32) -> Tuple[float, float]:
        self.train()
        epoch_loss = 0
        correct_predictions = 0
        total_samples = 0

        first_modality_data = next(iter(data.values()))
        num_batches = (len(first_modality_data) + batch_size - 1) // batch_size
        for i, (batch_data, batch_labels) in enumerate(self._batch_data(data, labels, batch_size)):
            print(f"\rTraining: {i+1}/{num_batches}", end="", flush=True)
            optimizer.zero_grad()
            outputs = self.forward(batch_data)
            loss = criterion(outputs, batch_labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Log gradients before update
            for name, param in self.named_parameters():
                if param.grad is not None:
                    logger.debug(f"Gradient norm for {name}: {param.grad.norm().item():.4f}")

            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == batch_labels).sum().item()
            total_samples += batch_labels.size(0)

            # Log batch-level metrics
            logger.debug(f"Batch loss: {loss.item():.4f}, Accuracy: {(predicted == batch_labels).sum().item() / batch_labels.size(0):.4f}")

        avg_loss = epoch_loss / num_batches
        accuracy = correct_predictions / total_samples

        logger.info(f"Epoch completed - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        return avg_loss, accuracy

    def _validate(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, criterion: nn.Module, batch_size: int = 32) -> Tuple[float, float]:
        self.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0

        first_modality_data = next(iter(data.values()))
        num_batches = (len(first_modality_data) + batch_size - 1) // batch_size

        with torch.no_grad():
            for i, (batch_data, batch_labels) in enumerate(self._batch_data(data, labels, batch_size)):
                print(f"\rValidating: {i+1}/{num_batches}", end="", flush=True)
                outputs = self.forward(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        avg_loss = val_loss / num_batches
        accuracy = correct_predictions / total_samples
        print()  # New line after progress indicator
        return avg_loss, accuracy

    def _batch_data(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, batch_size: int = 32):
        """Generate batches from the data."""
        num_samples = labels.size(0)
        for i in range(0, num_samples, batch_size):
            batch_data = {k: v[i:i+batch_size] for k, v in data.items()}
            batch_labels = labels[i:i+batch_size]
            yield batch_data, batch_labels

    def _save_best_model(self):
        """Save the best model state."""
        torch.save(self.state_dict(), 'best_model.pth')

    def _load_best_model(self):
        """Load the best model state."""
        self.load_state_dict(torch.load('best_model.pth'))

    def _update_performance(self, new_performance: float):
        """Update the performance history and trigger self-healing if necessary."""
        self.performance = new_performance
        self.performance_history.append(new_performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        self.last_update = time.time()

    def _self_heal(self):
        """Implement self-healing mechanisms with improved consistency and strategy selection."""
        logger.info("Initiating self-healing process...")
        initial_performance = self.performance
        initial_learning_rate = self.learning_rate
        best_performance = initial_performance
        best_learning_rate = initial_learning_rate
        improvement_threshold = 0.005  # Minimum improvement to consider a strategy successful
        strategy_effectiveness = {strategy.__name__: 0 for strategy in self.healing_strategies}

        logger.info(f"Initial state: Performance = {initial_performance:.4f}, Learning Rate = {initial_learning_rate:.6f}")

        for attempt in range(self.max_healing_attempts):
            # Select strategy based on past effectiveness
            strategy = max(self.healing_strategies, key=lambda s: strategy_effectiveness[s.__name__])
            strategy_name = strategy.__name__
            logger.info(f"Attempt {attempt + 1}: Applying strategy '{strategy_name}'")

            previous_lr = self.learning_rate
            previous_performance = self.performance
            strategy()
            new_performance = self._simulate_performance()

            logger.info(f"Attempt {attempt + 1}: "
                        f"LR change {previous_lr:.6f} -> {self.learning_rate:.6f}, "
                        f"Performance change {previous_performance:.4f} -> {new_performance:.4f}")

            # Update strategy effectiveness
            improvement = new_performance - previous_performance
            strategy_effectiveness[strategy_name] += improvement

            if new_performance > best_performance + improvement_threshold:
                best_performance = new_performance
                best_learning_rate = self.learning_rate
                logger.info(f"New best performance: {best_performance:.4f} (LR: {best_learning_rate:.6f})")

            if new_performance >= self.performance_threshold:
                logger.info(f"Self-healing successful after {attempt + 1} attempts.")
                self.performance = new_performance
                logger.info(f"Final state: Performance = {self.performance:.4f}, Learning Rate = {self.learning_rate:.6f}")
                return

        if best_performance > initial_performance + improvement_threshold:
            logger.info(f"Self-healing improved performance. "
                        f"Initial: {initial_performance:.4f} -> Final: {best_performance:.4f}")
            self.performance = best_performance
            self.learning_rate = best_learning_rate
        else:
            logger.warning("Self-healing not significantly improving performance. Reverting changes.")
            logger.info(f"Best achieved: {best_performance:.4f}, Reverting to: {initial_performance:.4f}")
            self.performance = initial_performance
            self.learning_rate = initial_learning_rate
            self._revert_model_changes()

        self._update_performance(self.performance)
        logger.info(f"Final state: Performance = {self.performance:.4f}, Learning Rate = {self.learning_rate:.6f}")
        logger.info(f"Strategy effectiveness: {strategy_effectiveness}")

    def _adjust_learning_rate(self):
        self.learning_rate *= (1 + LEARNING_RATE_ADJUSTMENT)
        self.learning_rate = min(self.learning_rate, 0.1)  # Cap at 0.1
        logger.info(f"Adjusted learning rate to {self.learning_rate:.6f}")

    def _reinitialize_layers(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        logger.info("Reinitialized model layers")

    def _increase_model_capacity(self):
        for name, modality in self.modalities.items():
            old_encoder = modality['encoder']
            new_encoder = nn.Sequential(
                old_encoder,
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
            self.modalities[name]['encoder'] = new_encoder
        logger.info("Increased model capacity")

    def _apply_regularization(self):
        for param in self.parameters():
            param.data += torch.randn_like(param) * 0.01
        logger.info("Applied noise regularization")

    def _revert_model_changes(self):
        try:
            self.load_state_dict(torch.load('best_model.pth'))
            logger.info("Reverted model to best known state")
        except FileNotFoundError:
            logger.warning("Could not find 'best_model.pth'. Unable to revert model changes.")
            logger.info("Continuing with current model state")

    def diagnose(self) -> List[str]:
        """Diagnose potential issues with the model."""
        issues = []
        if self.performance < self.performance_threshold:
            issues.append(f"Low performance: {self.performance:.4f}")
        if (time.time() - self.last_update) > self.update_interval:
            issues.append(f"Long time since last update: {(time.time() - self.last_update) / 3600:.2f} hours")
        if len(self.performance_history) >= 5 and all(p < self.performance_threshold for p in self.performance_history[-5:]):
            issues.append("Consistently low performance")
        if any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.parameters()):
            issues.append("NaN or Inf values detected in model parameters")
        return issues

    def adjust_learning_rate(self):
        """Adjust the learning rate based on recent performance."""
        if len(self.performance_history) >= 2:
            if self.performance_history[-1] > self.performance_history[-2]:
                self.learning_rate *= (1 + LEARNING_RATE_ADJUSTMENT)
            else:
                self.learning_rate *= (1 - LEARNING_RATE_ADJUSTMENT)
        self.learning_rate = max(min(self.learning_rate, 0.1), 1e-5)
        logger.info(f"Adjusted learning rate to {self.learning_rate:.6f}")

    def _simulate_performance(self) -> float:
        """Simulate new performance after applying healing strategies."""
        # Simulate performance improvement based on learning rate, current performance, and model state
        base_improvement = (1 - self.performance) * 0.1  # Base improvement factor
        lr_factor = np.sqrt(self.learning_rate / 0.001)  # Scale factor based on learning rate

        # Factor in model complexity
        model_complexity = sum(p.numel() for p in self.parameters()) / 1000000  # Normalize by million parameters
        complexity_factor = np.tanh(model_complexity)  # Bounded between 0 and 1

        # Consider recent performance trend
        recent_trend = 0
        if len(self.performance_history) >= 3:
            recent_trend = (self.performance_history[-1] - self.performance_history[-3]) / 3
        trend_factor = 1 + recent_trend

        improvement = base_improvement * lr_factor * complexity_factor * trend_factor
        noise = np.random.normal(0, 0.005 * lr_factor)  # Scaled noise

        new_performance = self.performance + improvement + noise
        return max(0.0, min(1.0, new_performance))  # Ensure performance is between 0 and 1

    def _plot_training_history(self):
        """Plot the training and validation loss history."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.plot(self.train_loss_history, label='Training Loss')
        plt.plot(self.val_loss_history, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss History')
        plt.legend()
        plt.savefig('loss_history.png')
        plt.close()

        plt.figure(figsize=(10, 5))
        plt.plot(self.performance_history, label='Performance')
        plt.axhline(y=self.performance_threshold, color='r', linestyle='--', label='Performance Threshold')
        plt.xlabel('Update')
        plt.ylabel('Performance')
        plt.title('Performance History')
        plt.legend()
        plt.savefig('performance_history.png')
        plt.close()

        logger.info("Training history plots saved as 'loss_history.png' and 'performance_history.png'")

# Example usage
if __name__ == "__main__":
    model = MultiModalLearning()
    model.add_modality('image', (3, 64, 64))
    model.add_modality('text', (100,))
    model.set_fusion_method('concatenation')

    # Simulated data
    batch_size = 32
    image_data = torch.randn(batch_size, 3, 64, 64)
    text_data = torch.randn(batch_size, 100)
    labels = torch.randint(0, 10, (batch_size,))

    inputs = {'image': image_data, 'text': text_data}
    fused_output = model.forward(inputs)
    print("Fused output shape:", fused_output.shape)

    model.train({'image': image_data, 'text': text_data}, labels)
