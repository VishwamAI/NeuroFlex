import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Optional
import optax
from sklearn.base import BaseEstimator, ClassifierMixin
from lale import operators as lale_ops
from art.attacks.evasion import FastGradientMethod
from art.experimental.estimators.classification import JaxClassifier
import torch
import torch.nn as nn
import torch.optim as optim
from NeuroFlex.core_neural_networks.data_loader import DataLoader
from NeuroFlex.core_neural_networks.tensorflow.tensorflow_module import create_tensorflow_model, train_tensorflow_model, tensorflow_predict
from NeuroFlex.core_neural_networks.pytorch.pytorch_module import pytorch_specific_function
from NeuroFlex.utils.utils import normalize_data, preprocess_data

class MachineLearning(nn.Module):
    features: List[int]
    activation: callable = nn.relu
    dropout_rate: float = 0.5
    use_lale: bool = False
    use_art: bool = False
    art_epsilon: float = 0.3
    use_tensorflow: bool = False

    @nn.compact
    def __call__(self, x, training: bool = False):
        if self.use_tensorflow:
            return self.tensorflow_forward(x, training)

        for feat in self.features[:-1]:
            x = nn.Dense(feat)(x)
            x = self.activation(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not training)(x)

        x = nn.Dense(self.features[-1])(x)

        if self.use_lale:
            x = self.lale_pipeline(x)

        if self.use_art and training:
            x = self.generate_adversarial_examples(x)

        return x

    def tensorflow_forward(self, x, training: bool = False):
        tf_model = create_tensorflow_model(input_shape=x.shape[1:], output_dim=self.features[-1], hidden_layers=self.features[:-1])
        return tensorflow_predict(tf_model, x)

    def setup_lale_pipeline(self):
        from lale.lib.sklearn import LogisticRegression, RandomForestClassifier

        classifier = lale_ops.make_choice(
            LogisticRegression, RandomForestClassifier
        )

        return classifier

    def generate_adversarial_examples(self, x):
        classifier = JaxClassifier(
            model=lambda x: self.apply({'params': self.params}, x),
            loss=lambda x, y: optax.softmax_cross_entropy(x, y),
            input_shape=x.shape[1:],
            nb_classes=self.features[-1]
        )

        attack = FastGradientMethod(classifier, eps=self.art_epsilon)
        x_adv = attack.generate(x)

        return x_adv

class NeuroFlexClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, features, activation=nn.relu, dropout_rate=0.5, learning_rate=0.001, use_tensorflow=False):
        self.features = features
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model = None
        self.params = None
        self.use_tensorflow = use_tensorflow

    def fit(self, X, y):
        self.model = MachineLearning(
            features=self.features,
            activation=self.activation,
            dropout_rate=self.dropout_rate,
            use_tensorflow=self.use_tensorflow
        )

        if self.use_tensorflow:
            tf_model = create_tensorflow_model(input_shape=X.shape[1:], output_dim=self.features[-1], hidden_layers=self.features[:-1])
            self.model = train_tensorflow_model(tf_model, X, y)
        else:
            key = jax.random.PRNGKey(0)
            params = self.model.init(key, X)

            optimizer = optax.adam(self.learning_rate)
            opt_state = optimizer.init(params)

            @jax.jit
            def train_step(params, opt_state, batch_x, batch_y):
                def loss_fn(params):
                    logits = self.model.apply({'params': params}, batch_x)
                    return optax.softmax_cross_entropy_with_integer_labels(logits, batch_y).mean()

                loss, grads = jax.value_and_grad(loss_fn)(params)
                updates, opt_state = optimizer.update(grads, opt_state)
                params = optax.apply_updates(params, updates)
                return params, opt_state, loss

            for epoch in range(100):  # Adjust number of epochs as needed
                params, opt_state, loss = train_step(params, opt_state, X, y)

            self.params = params

        return self

    def predict(self, X):
        if self.use_tensorflow:
            return tensorflow_predict(self.model, X)
        else:
            logits = self.model.apply({'params': self.params}, X)
            return jnp.argmax(logits, axis=-1)

    def predict_proba(self, X):
        if self.use_tensorflow:
            return tensorflow_predict(self.model, X)
        else:
            logits = self.model.apply({'params': self.params}, X)
            return nn.softmax(logits)

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
