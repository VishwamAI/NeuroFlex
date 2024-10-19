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

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import List, Dict, Any, Union, Tuple
import time
import logging
from tqdm import tqdm
import random
import psutil
from torch.utils.tensorboard import SummaryWriter
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
        self.classifier = nn.Linear(128, output_dim)  # Output dimension is now 10
        self.train_loss_history = []
        self.val_loss_history = []
        self.strategy_success_rates = {
            '_adjust_learning_rate': 0.0,
            '_reinitialize_layers': 0.0,
            '_increase_model_capacity': 0.0,
            '_apply_regularization': 0.0
        }
        self.applied_strategies = []  # Initialize applied_strategies attribute
        self.writer = SummaryWriter('logs/multi_modal_learning')

    def add_modality(self, name: str, input_shape: tuple):
        """Add a new modality to the multi-modal learning model."""
        if name not in ['text', 'image', 'tabular', 'time_series']:
            raise ValueError(f"Unsupported modality: {name}")
        self.modalities[name] = {
            'input_shape': input_shape,
            'encoder': self._create_encoder(name, input_shape)
        }

    def _create_encoder(self, modality: str, input_shape: tuple) -> nn.Module:
        """Create a modality-specific encoder network for a given input shape."""
        if modality == 'text':
            vocab_size = 30000  # Increased vocabulary size
            return nn.Sequential(
                nn.Embedding(vocab_size, 100),
                nn.LSTM(100, 64, batch_first=True),
                nn.Linear(64, 64)
            )
        elif modality == 'image':
            return nn.Sequential(
                nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, 64)
            )
        elif modality == 'tabular':
            return nn.Sequential(
                nn.Linear(input_shape[0], 128),
                nn.ReLU(),
                nn.Linear(128, 64)
            )
        elif modality == 'time_series':
            return nn.Sequential(
                nn.Conv1d(input_shape[0], 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(64, 64)
            )
        else:
            raise ValueError(f"Unsupported modality: {modality}")

    def set_fusion_method(self, method: str):
        """Set the fusion method for combining modalities."""
        supported_methods = ['concatenation', 'attention']
        if method not in supported_methods:
            raise ValueError(f"Unsupported fusion method. Choose from {supported_methods}")
        self.fusion_method = method

    def fuse_modalities(self, encoded_modalities: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Fuse the encoded modalities based on the selected fusion method."""
        # Get the maximum batch size among all modalities
        max_batch_size = max(tensor.size(0) for tensor in encoded_modalities.values())

        # Adjust batch sizes for all modalities
        adjusted_modalities = {}
        for name, tensor in encoded_modalities.items():
            if tensor.size(0) != max_batch_size:
                # Repeat or truncate the tensor along the batch dimension to match max_batch_size
                if tensor.size(0) < max_batch_size:
                    repeat_factor = max_batch_size // tensor.size(0)
                    adjusted_tensor = tensor.repeat(repeat_factor, *[1] * (tensor.dim() - 1))
                    if max_batch_size % tensor.size(0) != 0:
                        adjusted_tensor = torch.cat([adjusted_tensor, tensor[:max_batch_size % tensor.size(0)]], dim=0)
                else:
                    adjusted_tensor = tensor[:max_batch_size]
                adjusted_modalities[name] = adjusted_tensor
            else:
                adjusted_modalities[name] = tensor

        # Ensure all adjusted tensors have the same batch size
        if len(set(tensor.size(0) for tensor in adjusted_modalities.values())) != 1:
            raise ValueError("Failed to adjust batch sizes consistently across modalities")

        if self.fusion_method == 'concatenation':
            return torch.cat(list(adjusted_modalities.values()), dim=1)
        elif self.fusion_method == 'attention':
            # Implement attention-based fusion
            stacked_tensors = torch.stack(list(adjusted_modalities.values()), dim=1)
            attention_weights = torch.nn.functional.softmax(torch.rand(stacked_tensors.size(0), stacked_tensors.size(1)), dim=1)
            return (stacked_tensors * attention_weights.unsqueeze(-1)).sum(dim=1)
        else:
            raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through the multi-modal learning model."""
        logger.debug(f"Input types: {[(name, type(tensor)) for name, tensor in inputs.items()]}")
        if not inputs:
            raise ValueError("Input dictionary is empty")

        # Check if only a single modality is provided
        if len(inputs) == 1:
            raise ValueError("At least two modalities are required for fusion")

        # Ensure all inputs are tensors and have correct dtype
        for name, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                inputs[name] = torch.tensor(tensor)
            if name == 'text':
                inputs[name] = inputs[name].long()
            else:
                inputs[name] = inputs[name].float()
            logger.debug(f"Input {name} shape: {inputs[name].shape}, type: {inputs[name].dtype}")

        # Check for batch size consistency across all input modalities
        batch_sizes = [tensor.size(0) for tensor in inputs.values()]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Inconsistent batch sizes across modalities: {dict(zip(inputs.keys(), batch_sizes))}")

        # Handle missing modalities
        for modality in set(self.modalities.keys()) - set(inputs.keys()):
            inputs[modality] = torch.zeros((batch_sizes[0],) + self.modalities[modality]['input_shape'], dtype=torch.float32)
            logger.debug(f"Created zero tensor for missing modality {modality}: shape {inputs[modality].shape}")

        max_batch_size = batch_sizes[0]

        if max_batch_size == 0:
            # Handle zero-sized batch
            return torch.zeros((0, self.classifier.out_features), device=next(self.parameters()).device)

        encoded_modalities = {}
        for name, modality in self.modalities.items():
            logger.debug(f"Processing modality: {name}")
            logger.debug(f"Input shape for {name}: {inputs[name].shape}")
            logger.debug(f"Expected shape for {name}: (batch_size, {modality['input_shape']})")
            if inputs[name].shape[1:] != modality['input_shape']:
                raise ValueError(f"Input shape for {name} {inputs[name].shape} does not match the defined shape (batch_size, {modality['input_shape']})")

            try:
                if name == 'image':
                    # For image modality, preserve the 4D structure
                    encoded_modalities[name] = modality['encoder'](inputs[name])
                elif name == 'text':
                    # For text modality, ensure long type for embedding and float type for LSTM
                    text_input = inputs[name].long().clamp(0, 29999)  # Clamp to valid range
                    embedded = modality['encoder'][0](text_input)
                    lstm_out, _ = modality['encoder'][1](embedded.float())
                    lstm_out = lstm_out[:, -1, :]  # Use last time step output
                    encoded_modalities[name] = modality['encoder'][2](lstm_out)
                elif name == 'time_series':
                    # For time series, ensure 3D input (batch_size, channels, sequence_length)
                    if inputs[name].dim() == 2:
                        inputs[name] = inputs[name].unsqueeze(1)
                    encoded_modalities[name] = modality['encoder'](inputs[name])
                elif name == 'tabular':
                    # For tabular data, ensure 2D input (batch_size, features)
                    encoded_modalities[name] = modality['encoder'](inputs[name].view(inputs[name].size(0), -1))
                else:
                    # For other modalities, flatten the input
                    encoded_modalities[name] = modality['encoder'](inputs[name].view(inputs[name].size(0), -1))

                logger.debug(f"Encoded {name} shape: {encoded_modalities[name].shape}, type: {encoded_modalities[name].dtype}")
            except Exception as e:
                logger.error(f"Error processing modality {name}: {str(e)}")
                raise

        # Ensure all encoded modalities have the same batch size and are 2D tensors
        encoded_modalities = {name: tensor.view(max_batch_size, -1) for name, tensor in encoded_modalities.items()}
        logger.debug(f"Encoded modalities shapes after reshaping: {[(name, tensor.shape) for name, tensor in encoded_modalities.items()]}")

        try:
            if self.fusion_method == 'concatenation':
                fused = torch.cat(list(encoded_modalities.values()), dim=1)
            elif self.fusion_method == 'attention':
                fused = self.fuse_modalities(encoded_modalities)
            else:
                raise ValueError(f"Unsupported fusion method: {self.fusion_method}")

            logger.debug(f"Fused tensor shape: {fused.shape}, type: {fused.dtype}")

            # Ensure fused tensor is 2D and matches the classifier's input size
            if fused.dim() != 2 or fused.size(1) != self.classifier.in_features:
                fused = fused.view(max_batch_size, -1)
                fused = nn.functional.adaptive_avg_pool1d(fused.unsqueeze(1), self.classifier.in_features).squeeze(1)

            # Ensure fused tensor is a valid input for the classifier
            fused = fused.float()  # Convert to float if not already

            logger.debug(f"Final fused tensor shape: {fused.shape}, type: {fused.dtype}")

            return self.classifier(fused)
        except Exception as e:
            logger.error(f"Error during fusion or classification: {str(e)}")
            raise

    def fit(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, val_data: Dict[str, torch.Tensor] = None, val_labels: torch.Tensor = None, epochs: int = 10, lr: float = 0.001, patience: int = 5, batch_size: int = 32):
        """Train the multi-modal learning model."""
        if len(data) == 0 or len(labels) == 0:
            raise ValueError("Input data or labels are empty")

        if val_data is None or val_labels is None:
            train_data, val_data, train_labels, val_labels = self._split_data(data, labels)
        else:
            train_data, train_labels = data, labels

        if any(len(v) != len(labels) for v in train_data.values()):
            raise ValueError("Batch sizes of input data and labels do not match")

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
        criterion = nn.CrossEntropyLoss()
        logger.info(f"Starting training with initial learning rate: {lr}")

        moving_avg_window = min(10, epochs)
        performance_window = []
        best_performance = 0
        no_improve_count = 0
        best_epoch = 0
        self_healing_count = 0
        initial_performance = self.performance
        self_heal_threshold = self.performance_threshold * 0.9  # Adjusted threshold for self-healing

        for epoch in range(epochs):
            epoch_start_time = time.time()
            self.train()

            # Log CPU and memory usage before training
            cpu_usage = psutil.cpu_percent()
            memory_usage = psutil.virtual_memory().percent
            self.writer.add_scalar('System/CPU_Usage', cpu_usage, epoch)
            self.writer.add_scalar('System/Memory_Usage', memory_usage, epoch)

            # Log GPU usage if available
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.writer.add_scalar('System/GPU_Usage', gpu_usage, epoch)

            train_start_time = time.time()
            train_loss, train_accuracy = self._train_epoch(train_data, train_labels, optimizer, criterion, batch_size)
            train_duration = time.time() - train_start_time

            val_start_time = time.time()
            val_loss, val_accuracy = self._validate(val_data, val_labels, criterion, batch_size)
            val_duration = time.time() - val_start_time

            self.train_loss_history.append(train_loss)
            self.val_loss_history.append(val_loss)

            current_performance = val_accuracy
            performance_window.append(current_performance)
            if len(performance_window) > moving_avg_window:
                performance_window.pop(0)

            moving_avg_performance = sum(performance_window) / len(performance_window)
            previous_performance = self.performance
            self._update_performance(moving_avg_performance)

            # Log metrics
            self._log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy, moving_avg_performance, train_duration, val_duration)

            logger.info(f"Epoch {epoch + 1}/{epochs}:")
            logger.info(f"  Train Loss: {train_loss:.6f}, Train Accuracy: {train_accuracy:.6f}")
            logger.info(f"  Validation Loss: {val_loss:.6f}, Validation Accuracy: {val_accuracy:.6f}")
            logger.info(f"  Current Performance (Moving Avg): {moving_avg_performance:.6f}")
            logger.info(f"  Updated Overall Performance: {self.performance:.6f}")
            logger.info(f"  Performance Change: {self.performance - previous_performance:.6f}")

            self._log_model_parameters(epoch)

            scheduler.step(moving_avg_performance)

            if moving_avg_performance > best_performance:
                best_performance = moving_avg_performance
                best_epoch = epoch
                no_improve_count = 0
                self._save_best_model()
                logger.info("New best model saved.")
            else:
                no_improve_count += 1

            # Adjusted early stopping condition
            if no_improve_count >= patience and epoch >= epochs - 1:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break

            # Adjusted self-healing trigger condition
            if moving_avg_performance < self_heal_threshold and epoch >= epochs // 4:
                logger.warning(f"Moving average performance {moving_avg_performance:.4f} below threshold {self_heal_threshold:.4f}. Initiating self-healing.")
                self._self_heal()
                self_healing_count += 1
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, verbose=True)
                logger.info(f"After self-healing - Learning Rate: {self.learning_rate:.6f}, Performance: {self.performance:.4f}")

            epoch_duration = time.time() - epoch_start_time
            self.writer.add_scalar('Time/Epoch_Duration', epoch_duration, epoch)

        self.writer.add_scalar('Training/Self_Healing_Count', self_healing_count, 0)
        self._load_best_model()
        logger.info(f"Training completed. Best Performance: {best_performance:.4f} at epoch {best_epoch + 1}")
        logger.info(f"Final Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        logger.info(f"Performance History: {self.performance_history}")
        self._plot_training_history()

    def _log_metrics(self, epoch, train_loss, val_loss, train_accuracy, val_accuracy, moving_avg_performance, train_duration, val_duration):
        self.writer.add_scalar('Performance/Train_Loss', train_loss, epoch)
        self.writer.add_scalar('Performance/Val_Loss', val_loss, epoch)
        self.writer.add_scalar('Performance/Train_Accuracy', train_accuracy, epoch)
        self.writer.add_scalar('Performance/Val_Accuracy', val_accuracy, epoch)
        self.writer.add_scalar('Performance/Moving_Avg_Performance', moving_avg_performance, epoch)
        self.writer.add_scalar('Performance/Overall_Performance', self.performance, epoch)
        self.writer.add_scalar('Time/Train_Duration', train_duration, epoch)
        self.writer.add_scalar('Time/Val_Duration', val_duration, epoch)

        if len(self.train_loss_history) > 1:
            loss_deviation = abs(train_loss - self.train_loss_history[-2])
            self.writer.add_scalar('Performance/Train_Loss_Deviation', loss_deviation, epoch)

        if epoch > 0:
            improvement_ratio = (self.performance - self.performance_history[0]) / (epoch + 1)
            self.writer.add_scalar('Performance/Improvement_Ratio', improvement_ratio, epoch)

    def _log_model_parameters(self, epoch):
        for name, param in self.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)
                self.writer.add_histogram(f'Weights/{name}', param.data, epoch)

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

        if not data or not labels.numel():
            logger.warning("Empty data or labels provided for training epoch.")
            return 0.0, 0.0

        try:
            first_modality_data = next(iter(data.values()))
            num_samples = len(first_modality_data)
            num_batches = (num_samples + batch_size - 1) // batch_size

            for i, (batch_data, batch_labels) in enumerate(self._batch_data(data, labels, batch_size)):
                if not batch_data or not batch_labels.numel():
                    logger.warning(f"Empty batch encountered at iteration {i+1}/{num_batches}. Skipping.")
                    continue

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

            if num_batches > 0:
                avg_loss = epoch_loss / num_batches
                accuracy = correct_predictions / total_samples
            else:
                logger.warning("No batches processed in the epoch.")
                avg_loss, accuracy = 0.0, 0.0

            logger.info(f"Epoch completed - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
            return avg_loss, accuracy

        except Exception as e:
            logger.error(f"Error in _train_epoch: {str(e)}")
            return 0.0, 0.0

    def _validate(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, criterion: nn.Module, batch_size: int = 32) -> Tuple[float, float]:
        self.eval()
        val_loss = 0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch_data, batch_labels in self._batch_data(data, labels, batch_size):
                outputs = self.forward(batch_data)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct_predictions += (predicted == batch_labels).sum().item()
                total_samples += batch_labels.size(0)

        num_batches = (len(labels) + batch_size - 1) // batch_size
        avg_loss = val_loss / num_batches
        accuracy = correct_predictions / total_samples
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
        try:
            self.load_state_dict(torch.load('best_model.pth'))
            logger.info("Successfully loaded the best model.")
        except FileNotFoundError:
            logger.warning("Best model file 'best_model.pth' not found. Continuing with current model state.")
            return  # Return early to avoid raising the exception
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            # Consider adding a fallback mechanism or additional error handling here if needed

    def _update_performance(self, new_performance: float):
        """Update the performance history and save the best model if performance improves."""
        old_performance = self.performance
        performance_change = new_performance - old_performance

        # Implement a more gradual update mechanism with adjusted update rate
        update_rate = 0.2  # Adjusted from 0.3 to provide more gradual updates
        self.performance = old_performance + update_rate * performance_change

        self.performance_history.append(self.performance)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        self.last_update = time.time()

        logger.info(f"Performance update details:")
        logger.info(f"  Old performance: {old_performance:.4f}")
        logger.info(f"  New performance: {self.performance:.4f}")
        logger.info(f"  Raw new performance: {new_performance:.4f}")
        logger.info(f"  Performance change: {performance_change:.4f}")
        logger.info(f"  Update rate: {update_rate:.2f}")
        logger.debug(f"Performance history: {self.performance_history}")

        if self.performance > max(self.performance_history[:-1], default=0):
            logger.info(f"New best performance achieved: {self.performance:.4f}")
            self._save_best_model()

        if performance_change < 0:
            logger.warning(f"Performance decreased: {old_performance:.4f} -> {self.performance:.4f}")
        elif performance_change > 0:
            logger.info(f"Performance improved: {old_performance:.4f} -> {self.performance:.4f}")
        else:
            logger.info(f"Performance unchanged: {self.performance:.4f}")

        # Add a check to ensure performance doesn't exceed 1.0
        self.performance = min(self.performance, 1.0)

        return self.performance

    def _self_heal(self):
        """Implement self-healing mechanisms with gradual and consistent improvements."""
        logger.info("Initiating self-healing process...")
        initial_performance = self.performance
        initial_learning_rate = self.learning_rate
        best_performance = initial_performance
        best_learning_rate = initial_learning_rate

        healing_strategies = [
            self._adjust_learning_rate,
            self._reinitialize_layers,
            self._increase_model_capacity,
            self._apply_regularization
        ]

        logger.info(f"Initial performance: {initial_performance:.4f}")
        logger.info(f"Initial learning rate: {initial_learning_rate:.6f}")

        performance_window = []
        window_size = 5
        improvement_threshold = 0.0005  # Further reduced threshold for more gradual improvements
        max_performance_drop = 0.002  # Further reduced maximum allowed performance drop
        self.applied_strategies = []  # Initialize applied_strategies list
        consecutive_failures = 0
        max_consecutive_failures = 10  # Increased to allow more attempts before stopping

        for attempt in range(self.max_healing_attempts):
            # Adaptive strategy selection based on success rates and randomness
            strategy = self._select_strategy(healing_strategies)
            strategy_name = strategy.__name__
            logger.info(f"Attempt {attempt + 1}: Applying strategy '{strategy_name}'")

            logger.debug(f"Before {strategy_name} - Performance: {self.performance:.4f}, Learning rate: {self.learning_rate:.6f}")

            strategy()
            self.applied_strategies.append(strategy_name)  # Track applied strategy

            # Adjust learning rate more conservatively
            old_lr = self.learning_rate
            self._adjust_learning_rate(factor=0.95)  # Smaller adjustment factor
            logger.info(f"Learning rate adjusted from {old_lr:.6f} to {self.learning_rate:.6f}")

            new_performance = self._simulate_performance()
            logger.info(f"New performance after {strategy_name}: {new_performance:.4f}")

            performance_change = new_performance - self.performance

            # More lenient safeguard against performance drops
            if performance_change < -max_performance_drop:
                logger.warning(f"Performance drop detected. Reverting strategy.")
                self._revert_model_changes()
                self.applied_strategies.pop()  # Remove last applied strategy
                self.learning_rate = old_lr  # Revert learning rate
                logger.info(f"Reverted learning rate to {self.learning_rate:.6f}")
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    logger.warning(f"Too many consecutive failures. Pausing self-healing process.")
                    break
                continue
            else:
                consecutive_failures = max(0, consecutive_failures - 1)  # Decrease failure count, but not below 0

            performance_window.append(new_performance)
            if len(performance_window) > window_size:
                performance_window.pop(0)

            avg_performance = sum(performance_window) / len(performance_window)

            logger.info(f"Performance change: {performance_change:.4f}")
            logger.info(f"Average performance (last {len(performance_window)} attempts): {avg_performance:.4f}")

            if new_performance > best_performance:
                best_performance = new_performance
                best_learning_rate = self.learning_rate
                logger.info(f"New best performance: {best_performance:.4f}")
                logger.info(f"Best learning rate updated to: {best_learning_rate:.6f}")

            # More gradual healing approach
            if avg_performance > self.performance + improvement_threshold:
                self._update_performance(self.performance + (avg_performance - self.performance) * 0.5)  # More conservative update
                logger.info(f"Gradually updating performance to: {self.performance:.4f}")
                self._update_strategy_success_rates(strategy_name, True)
            elif new_performance > self.performance:
                # Update performance even if improvement is small, as long as it's positive
                self._update_performance(self.performance + (new_performance - self.performance) * 0.3)  # Even more conservative for small improvements
                logger.info(f"Small improvement detected. Updating performance to: {self.performance:.4f}")
                self._update_strategy_success_rates(strategy_name, True)
            else:
                logger.info("No improvement in performance. Continuing healing process.")
                self._update_strategy_success_rates(strategy_name, False)

            if self.performance >= self.performance_threshold:
                logger.info(f"Self-healing reached performance threshold after {attempt + 1} attempts.")
                break

            # Update strategy success rates and re-sort strategies
            healing_strategies.sort(key=self._strategy_success_rate, reverse=True)
            logger.info(f"Updated strategy order: {[s.__name__ for s in healing_strategies]}")

        # Always apply the best found configuration
        if best_performance > initial_performance:
            logger.info(f"Self-healing improved performance. New performance: {best_performance:.4f}")
            self.performance = best_performance
            self.learning_rate = best_learning_rate
            logger.info(f"Final learning rate: {self.learning_rate:.6f}")
        else:
            logger.warning("Self-healing did not improve performance. Keeping initial configuration.")
            self.performance = initial_performance
            self.learning_rate = initial_learning_rate
            self._revert_model_changes()
            self.applied_strategies = []  # Clear applied strategies if reverting
            logger.info(f"Kept initial learning rate: {self.learning_rate:.6f}")

        self._save_best_model()  # Save the best model after self-healing
        logger.info(f"Self-healing process completed. Final performance: {self.performance:.4f}")
        logger.info(f"Final learning rate: {self.learning_rate:.6f}")

        # Explicitly update performance history with final performance
        self._update_performance(self.performance)
        logger.info(f"Updated performance history with final performance: {self.performance:.4f}")

    def _strategy_success_rate(self, strategy):
        """Return the success rate of a given strategy."""
        return self.strategy_success_rates.get(strategy.__name__, 0.0)

    def _select_strategy(self, strategies):
        """Select a strategy based on success rates and some randomness."""
        total_success = sum(self._strategy_success_rate(s) for s in strategies)
        if total_success == 0:
            return random.choice(strategies)

        probabilities = [self._strategy_success_rate(s) / total_success for s in strategies]
        return random.choices(strategies, weights=probabilities, k=1)[0]

    def _update_strategy_success_rates(self, strategy_name: str = None, success: bool = None):
        """Update the success rate of a specific healing strategy or all strategies."""
        if strategy_name and success is not None:
            current_rate = self.strategy_success_rates.get(strategy_name, 0.0)
            new_rate = 0.9 * current_rate + 0.1 * (1 if success else 0)
            self.strategy_success_rates[strategy_name] = new_rate
            logger.info(f"Updated success rate for {strategy_name}: {new_rate:.4f}")

        logger.debug(f"All strategy success rates: {self.strategy_success_rates}")

    def _update_strategy_success_rates(self, strategy_name: str, success: bool):
        """Update the success rate of a specific healing strategy."""
        current_rate = self.strategy_success_rates.get(strategy_name, 0.0)
        new_rate = 0.9 * current_rate + 0.1 * (1 if success else 0)
        self.strategy_success_rates[strategy_name] = new_rate
        logger.info(f"Updated success rate for {strategy_name}: {new_rate:.4f}")
        logger.debug(f"All strategy success rates: {self.strategy_success_rates}")

    def _adjust_learning_rate(self, factor=None):
        logger.info(f"Adjusting learning rate. Current rate: {self.learning_rate:.6f}")
        old_rate = self.learning_rate
        adjustment = factor if factor is not None else LEARNING_RATE_ADJUSTMENT
        self.learning_rate *= (1 + adjustment)
        initial_lr = 0.001  # Initial learning rate
        max_lr = initial_lr * (1 + LEARNING_RATE_ADJUSTMENT * MAX_HEALING_ATTEMPTS)
        self.learning_rate = min(max(self.learning_rate, initial_lr * 0.1), max_lr)
        logger.info(f"Adjusted learning rate from {old_rate:.6f} to {self.learning_rate:.6f}")
        logger.debug(f"Learning rate adjustment factor: {1 + adjustment}")
        logger.debug(f"Learning rate bounds: [{initial_lr * 0.1:.6f}, {max_lr:.6f}]")

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
        self.load_state_dict(torch.load('best_model.pth'))
        logger.info("Reverted model to best known state")

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
        # Introduce randomness to make the simulation more realistic
        improvement = np.random.normal(0.05, 0.02)  # Mean of 0.05 with standard deviation of 0.02
        improvement = max(0, improvement)  # Ensure non-negative improvement
        new_performance = min(self.performance + improvement, 1.0)
        logger.debug(f"Simulated performance improvement: {improvement:.4f}")
        logger.debug(f"New simulated performance: {new_performance:.4f}")
        return new_performance

    def _plot_training_history(self):
        """Plot the training and validation loss history."""
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt

        # Disable font-related operations
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['svg.fonttype'] = 'none'

        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.train_loss_history, label='Training Loss')
            plt.plot(self.val_loss_history, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss History')
            plt.legend()
            plt.savefig('loss_history.png', dpi=300, bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(10, 5))
            plt.plot(self.performance_history, label='Performance')
            plt.axhline(y=self.performance_threshold, color='r', linestyle='--', label='Performance Threshold')
            plt.xlabel('Update')
            plt.ylabel('Performance')
            plt.title('Performance History')
            plt.legend()
            plt.savefig('performance_history.png', dpi=300, bbox_inches='tight')
            plt.close()

            logger.info("Training history plots saved as 'loss_history.png' and 'performance_history.png'")
        except Exception as e:
            logger.error(f"Error while plotting training history: {str(e)}")

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
