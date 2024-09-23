import tensorflow as tf
from typing import List, Tuple, Optional
import time
import logging


class TensorFlowModel(tf.keras.Model):
    def __init__(
        self, input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]
    ):
        super(TensorFlowModel, self).__init__()
        self.input_dim = input_shape[0]  # Assuming 1D input
        self.layers_list = []

        # Input layer
        self.layers_list.append(
            tf.keras.layers.Dense(
                hidden_layers[0], activation="relu", input_shape=input_shape
            )
        )

        # Hidden layers
        for i in range(1, len(hidden_layers)):
            self.layers_list.append(
                tf.keras.layers.Dense(hidden_layers[i], activation="relu")
            )

        # Output layer
        self.layers_list.append(tf.keras.layers.Dense(output_dim))

        self.model = tf.keras.Sequential(self.layers_list)

        # Self-healing attributes
        self.performance = 0.0
        self.last_update = time.time()
        self.performance_history = []
        self.performance_threshold = 0.8
        self.update_interval = 86400  # 24 hours in seconds
        self.learning_rate = 0.001

    def call(self, x: tf.Tensor) -> tf.Tensor:
        return self.model(x)


def create_tensorflow_model(
    input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]
) -> TensorFlowModel:
    return TensorFlowModel(input_shape, output_dim, hidden_layers)


def train_tensorflow_model(
    model: TensorFlowModel,
    x_train: tf.Tensor,
    y_train: tf.Tensor,
    epochs: int = 10,
    batch_size: int = 32,
    validation_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None,
) -> dict:
    optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
    loss_fn = tf.keras.losses.MeanSquaredError()
    train_loss = tf.keras.metrics.Mean(name="train_loss")
    val_loss = tf.keras.metrics.Mean(name="val_loss")
    history = {"train_loss": [], "val_loss": []}

    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step(x, y):
        predictions = model(x, training=False)
        loss = loss_fn(y, predictions)
        val_loss(loss)

    for epoch in range(epochs):
        train_loss.reset_states()
        val_loss.reset_states()

        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i : i + batch_size]
            batch_y = y_train[i : i + batch_size]
            train_step(batch_x, batch_y)

        if validation_data:
            val_x, val_y = validation_data
            test_step(val_x, val_y)

        history["train_loss"].append(train_loss.result().numpy())
        if validation_data:
            history["val_loss"].append(val_loss.result().numpy())

        # Self-healing mechanism
        model.performance = (
            1.0 - train_loss.result().numpy()
        )  # Assuming lower loss is better
        model.performance_history.append(model.performance)
        if len(model.performance_history) > 100:
            model.performance_history.pop(0)

        if model.performance < model.performance_threshold:
            logging.info(
                f"Performance below threshold. Current: {model.performance:.4f}"
            )
            model.learning_rate *= 0.9
            optimizer.learning_rate.assign(model.learning_rate)
            logging.info(f"Adjusted learning rate to {model.learning_rate:.6f}")

        model.last_update = time.time()

    return history


def tensorflow_predict(model: TensorFlowModel, x: tf.Tensor) -> tf.Tensor:
    return model(x, training=False)


def diagnose(model: TensorFlowModel) -> List[str]:
    issues = []
    if model.performance < model.performance_threshold:
        issues.append(f"Low performance: {model.performance:.4f}")
    if (time.time() - model.last_update) > model.update_interval:
        issues.append(
            f"Long time since last update: {(time.time() - model.last_update) / 3600:.2f} hours"
        )
    if len(model.performance_history) > 5 and all(
        p < model.performance_threshold for p in model.performance_history[-5:]
    ):
        issues.append("Consistently low performance")
    return issues


def self_heal(model: TensorFlowModel, x_train: tf.Tensor, y_train: tf.Tensor):
    issues = diagnose(model)
    if issues:
        logging.info(f"Self-healing triggered. Issues: {issues}")
        model.learning_rate *= 1.1  # Increase learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=model.learning_rate)
        loss_fn = tf.keras.losses.MeanSquaredError()

        # Perform a quick retraining
        for _ in range(5):  # 5 quick epochs
            with tf.GradientTape() as tape:
                predictions = model(x_train, training=True)
                loss = loss_fn(y_train, predictions)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        model.performance = 1.0 - loss.numpy()
        model.performance_history.append(model.performance)
        model.last_update = time.time()
        logging.info(f"Self-healing complete. New performance: {model.performance:.4f}")
