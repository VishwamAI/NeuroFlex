import tensorflow as tf
from typing import List, Tuple, Optional

class TensorFlowConvolutions(tf.keras.layers.Layer):
    def __init__(self, filters: List[int], kernel_sizes: List[Tuple[int, int]], strides: List[Tuple[int, int]],
                 paddings: List[str], activation: str = 'relu'):
        super(TensorFlowConvolutions, self).__init__()
        self.conv_layers = []
        for f, k, s, p in zip(filters, kernel_sizes, strides, paddings):
            self.conv_layers.append(tf.keras.layers.Conv2D(filters=f, kernel_size=k, strides=s, padding=p, activation=activation))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = inputs
        for layer in self.conv_layers:
            x = layer(x)
        return self.flatten(x)

def create_conv_model(input_shape: Tuple[int, ...], filters: List[int], kernel_sizes: List[Tuple[int, int]],
                      strides: List[Tuple[int, int]], paddings: List[str], output_dim: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=input_shape)
    conv_layers = TensorFlowConvolutions(filters, kernel_sizes, strides, paddings)
    x = conv_layers(inputs)
    outputs = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def train_conv_model(model: tf.keras.Model, x_train: tf.Tensor, y_train: tf.Tensor,
                     epochs: int = 10, batch_size: int = 32,
                     validation_data: Optional[Tuple[tf.Tensor, tf.Tensor]] = None) -> tf.keras.callbacks.History:
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

def conv_predict(model: tf.keras.Model, x: tf.Tensor) -> tf.Tensor:
    return model(x)
