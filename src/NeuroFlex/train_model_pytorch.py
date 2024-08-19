# Import required modules.
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import logging

class CustomDataset(Dataset):
    """Custom dataset class for demonstration."""
    def __init__(self, data_tensor, label_tensor):
        self.data_tensor = data_tensor
        self.label_tensor = label_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return self.data_tensor[idx], self.label_tensor[idx]

def create_data_loaders(train_data, train_labels, val_data, val_labels, batch_size):
    """Create DataLoader objects for training and validation datasets."""
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, num_epochs=20, learning_rate=1e-3):
    """Train the specified PyTorch model."""
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        logging.info(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss/len(train_loader)}")
        validate_model(model, val_loader)

def validate_model(model, val_loader):
    """Evaluate model performance on the validation set."""
    model.eval()
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    val_accuracy = correct_predictions / total_predictions
    logging.info(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
