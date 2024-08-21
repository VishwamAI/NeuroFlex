import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PyTorchModel(nn.Module):
    def __init__(self, features):
        super(PyTorchModel, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(len(features) - 1):
            self.layers.append(nn.Linear(features[i], features[i+1]))
            if i < len(features) - 2:
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_pytorch_model(model, X, y, epochs=10, learning_rate=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model
