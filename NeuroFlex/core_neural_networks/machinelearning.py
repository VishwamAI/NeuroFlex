from typing import List, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from lale import operators as lale_ops
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from NeuroFlex.core_neural_networks.data_loader import DataLoader
from NeuroFlex.core_neural_networks.tensorflow.tensorflow_module import create_tensorflow_model, train_tensorflow_model, tensorflow_predict
from NeuroFlex.core_neural_networks.pytorch.pytorch_module import create_pytorch_model
from NeuroFlex.utils.utils import normalize_data, preprocess_data

class MachineLearning(torch.nn.Module):
    def __init__(self, features: List[int], activation: callable = torch.nn.ReLU(),
                 dropout_rate: float = 0.5, use_lale: bool = False,
                 use_art: bool = False, art_epsilon: float = 0.3):
        super(MachineLearning, self).__init__()
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_lale = use_lale
        self.use_art = use_art
        self.art_epsilon = art_epsilon

        self.layers = torch.nn.ModuleList()
        for i in range(len(features) - 1):
            self.layers.append(torch.nn.Linear(features[i], features[i+1]))
            if i < len(features) - 2:
                self.layers.append(self.activation)
                self.layers.append(torch.nn.Dropout(self.dropout_rate))

    def forward(self, x, training: bool = False):
        for layer in self.layers:
            if isinstance(layer, torch.nn.Dropout):
                x = layer(x, training)
            else:
                x = layer(x)

        if self.use_lale:
            x = self.lale_pipeline(x)

        if self.use_art and training:
            x = self.generate_adversarial_examples(x)

        return x

    def setup_lale_pipeline(self):
        from lale.lib.sklearn import LogisticRegression, RandomForestClassifier

        classifier = lale_ops.make_choice(
            LogisticRegression, RandomForestClassifier
        )

        return classifier

    def generate_adversarial_examples(self, x):
        classifier = PyTorchClassifier(
            model=self,
            loss=torch.nn.CrossEntropyLoss(),
            input_shape=x.shape[1:],
            nb_classes=self.features[-1]
        )

        attack = FastGradientMethod(classifier, eps=self.art_epsilon)
        x_adv = attack.generate(x)

        return x_adv

class NeuroFlexClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, features, activation=torch.nn.ReLU(), dropout_rate=0.5, learning_rate=0.001):
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None

    def fit(self, X, y):
        self.model = MachineLearning(
            features=self.features,
            activation=self.activation,
            dropout_rate=self.dropout_rate
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)

        for epoch in range(100):  # Adjust number of epochs as needed
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        return self

    def predict(self, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
        return torch.argmax(logits, dim=-1).numpy()

    def predict_proba(self, X):
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
        return torch.nn.functional.softmax(logits, dim=-1).numpy()

class PyTorchModel(torch.nn.Module):
    def __init__(self, features):
        super(PyTorchModel, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(len(features) - 1):
            self.layers.append(torch.nn.Linear(features[i], features[i+1]))
            if i < len(features) - 2:
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Dropout(0.5))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def train_pytorch_model(model, X, y, learning_rate=0.001, epochs=100):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.LongTensor(y)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    return model

# Additional utility functions for machine learning can be added here
