import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import logging

class PyTorchIntegration:
    def __init__(self, input_dim, hidden_dims, output_dim):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.model = self.define_model()

    def define_model(self):
        layers = []
        prev_dim = self.input_dim
        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, self.output_dim))
        return nn.Sequential(*layers)

    def train_model(self, train_data, train_labels, val_data, val_labels, num_epochs=20, batch_size=32, learning_rate=1e-3):
        train_dataset = TensorDataset(torch.FloatTensor(train_data), torch.LongTensor(train_labels))
        val_dataset = TensorDataset(torch.FloatTensor(val_data), torch.LongTensor(val_labels))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            val_accuracy = self.evaluate_model(val_loader)
            logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader):.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    def evaluate_model(self, data_loader):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in data_loader:
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return 100 * correct / total

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        logging.info(f"Model saved to {path}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
        logging.info(f"Model loaded from {path}")

    def predict(self, data):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.FloatTensor(data))
