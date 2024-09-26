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
import numpy as np
import tensorflow as tf
import torch

class ArrayLibraries:
    @staticmethod
    def jax_operations(x):
        # Basic JAX operations
        result = jax.numpy.sum(x)
        result = jax.numpy.mean(x, axis=0)
        result = jax.numpy.max(x)
        return result

    @staticmethod
    def numpy_operations(x):
        # Basic NumPy operations
        result = np.sum(x)
        result = np.mean(x, axis=0)
        result = np.max(x)
        return result

    @staticmethod
    def tensorflow_operations(x):
        # Basic TensorFlow operations
        result = tf.reduce_sum(x)
        result = tf.reduce_mean(x, axis=0)
        result = tf.reduce_max(x)
        return result

    @staticmethod
    def pytorch_operations(x):
        # Basic PyTorch operations
        result = torch.sum(x)
        result = torch.mean(x, dim=0)
        result = torch.max(x)
        return result

    @staticmethod
    def convert_jax_to_numpy(x):
        return np.array(x)

    @staticmethod
    def convert_numpy_to_jax(x):
        return jnp.array(x)

    @staticmethod
    def convert_numpy_to_tensorflow(x):
        return tf.convert_to_tensor(x)

    @staticmethod
    def convert_tensorflow_to_numpy(x):
        return x.numpy()

    @staticmethod
    def convert_numpy_to_pytorch(x):
        return torch.from_numpy(x)

    @staticmethod
    def convert_pytorch_to_numpy(x):
        return x.detach().cpu().numpy()
