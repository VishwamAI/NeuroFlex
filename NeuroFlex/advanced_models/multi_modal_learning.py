import numpy as np
import torch
import torch.nn as nn
from typing import List, Dict, Any, Union

class MultiModalLearning:
    def __init__(self):
        self.modalities = {}
        self.fusion_method = None

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
        return fused

    def train(self, data: Dict[str, torch.Tensor], labels: torch.Tensor, epochs: int = 10, lr: float = 0.001):
        """Train the multi-modal learning model."""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.forward(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

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
