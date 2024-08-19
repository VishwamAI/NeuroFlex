import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchModel(nn.Module):
    def __init__(self, features):
        super(PyTorchModel, self).__init__()
        self.layer = nn.Linear(features, features)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.layer(x)
        return self.activation(x)

def train_pytorch_model(model, X, y, epochs=1, learning_rate=0.001):
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

    return model.state_dict()
