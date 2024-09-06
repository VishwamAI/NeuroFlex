import tensorflow as tf
from typing import List, Tuple, Optional

class TensorFlowModel(tf.keras.Model):
    def __init__(self, input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]):
        super(TensorFlowModel, self).__init__()
        self.input_layer = tf.keras.layers.InputLayer(input_shape=input_shape)
        self.hidden_layers = [tf.keras.layers.Dense(units, activation='relu') for units in hidden_layers]
        self.output_layer = tf.keras.layers.Dense(output_dim)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)

def create_tensorflow_model(input_shape: Tuple[int, ...], output_dim: int, hidden_layers: List[int]) -> TensorFlowModel:
    return TensorFlowModel(input_shape, output_dim, hidden_layers)

def train_tensorflow_model(model: TensorFlowModel,
                           x_train: tf.Tensor,
                           y_train: tf.Tensor,
                           epochs: int = 10,
                           batch_size: int = 32,
                           validation_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.keras.callbacks.History:
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

def tensorflow_predict(model: TensorFlowModel, x: tf.Tensor) -> tf.Tensor:
    return model(x)
