# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import jax
import jax.numpy as jnp
import flax.linen as nn
from tensorflow import keras
import tensorflow as tf
from typing import Callable, Tuple



class GoogleIntegration:
    def __init__(self, input_shape, num_classes):
        if len(input_shape) != 3:
            raise ValueError("Input shape must be 3-dimensional (height, width, channels)")
        if num_classes <= 0:
            raise ValueError("Number of classes must be positive")
        self.input_shape = input_shape
        self.num_classes = num_classes

    def create_cnn_model(self) -> nn.Module:
        class CNN(nn.Module):
            num_classes: int

            @nn.compact
            def __call__(self, x):
                x = nn.Conv(features=32, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = nn.Conv(features=64, kernel_size=(3, 3))(x)
                x = nn.relu(x)
                x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
                x = x.reshape((x.shape[0], -1))  # flatten
                x = nn.Dense(features=256)(x)
                x = nn.relu(x)
                x = nn.Dense(features=self.num_classes)(x)
                return x

        return CNN(num_classes=self.num_classes)





    def xla_compilation(self, model: nn.Module, input_shape: tuple) -> Callable:
        @jax.jit
        def forward(variables, inputs):
            return model.apply(variables, inputs)

        return forward

    def integrate_tensorflow_model(self, tf_model: keras.Model) -> nn.Module:
        class TFWrapper(nn.Module):
            @nn.compact
            def __call__(self, x):
                # Convert JAX array to TensorFlow tensor
                x_tf = tf.convert_to_tensor(x)
                # Run TensorFlow model
                y_tf = tf_model(x_tf)
                # Convert back to JAX array
                return jnp.array(y_tf)

        return TFWrapper()

# Example usage:
# google_integration = GoogleIntegration((28, 28, 1), 10)
# cnn_model = google_integration.create_cnn_model()
# compiled_cnn = google_integration.xla_compilation(cnn_model, (1, 28, 28, 1))
