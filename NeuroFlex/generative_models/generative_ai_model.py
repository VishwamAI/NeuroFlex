import torch
import torch.nn as nn

class GenerativeAIModel(nn.Module):
    def __init__(self, input_dim=100, output_dim=784):
        super(GenerativeAIModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)

    def generate(self, num_samples=1, device='cpu'):
        z = torch.randn(num_samples, 100).to(device)
        return self.forward(z)
