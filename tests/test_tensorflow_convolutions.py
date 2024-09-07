import pytest
import tensorflow as tf
import numpy as np
from NeuroFlex.core_neural_networks.tensorflow.tensorflow_convolutions import TensorFlowConvolutions
from NeuroFlex.utils import convert_array

@pytest.fixture
def conv2d_config():
    return {
        'features': (32, 64, 128),
        'input_shape': (1, 28, 28, 1),
        'conv_dim': 2
    }

@pytest.fixture
def conv3d_config():
    return {
        'features': (16, 32, 64),
        'input_shape': (1, 16, 16, 16, 1),
        'conv_dim': 3
    }

def test_tensorflow_convolutions_init(conv2d_config):
    tf_conv = TensorFlowConvolutions(**conv2d_config)
    assert tf_conv.features == conv2d_config['features']
    assert tf_conv.input_shape == conv2d_config['input_shape']
    assert tf_conv.conv_dim == conv2d_config['conv_dim']

def test_conv2d_block(conv2d_config):
    tf_conv = TensorFlowConvolutions(**conv2d_config)
    input_tensor = tf.random.normal(conv2d_config['input_shape'])
    output = tf_conv.conv2d_block(input_tensor)
    assert isinstance(output, tf.Tensor)
    assert output.shape.rank == 2  # Flattened output

def test_conv3d_block(conv3d_config):
    tf_conv = TensorFlowConvolutions(**conv3d_config)
    input_tensor = tf.random.normal(conv3d_config['input_shape'])
    output = tf_conv.conv3d_block(input_tensor)
    assert isinstance(output, tf.Tensor)
    assert output.shape.rank == 2  # Flattened output

def test_create_model_2d(conv2d_config):
    tf_conv = TensorFlowConvolutions(**conv2d_config)
    model = tf_conv.create_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None,) + conv2d_config['input_shape'][1:]
    assert model.output_shape == (None, conv2d_config['features'][-1])

def test_create_model_3d(conv3d_config):
    tf_conv = TensorFlowConvolutions(**conv3d_config)
    model = tf_conv.create_model()
    assert isinstance(model, tf.keras.Model)
    assert model.input_shape == (None,) + conv3d_config['input_shape'][1:]
    assert model.output_shape == (None, conv3d_config['features'][-1])

def test_invalid_conv_dim():
    with pytest.raises(ValueError, match="conv_dim must be 2 or 3"):
        TensorFlowConvolutions(features=(32, 64), input_shape=(1, 28, 28, 1), conv_dim=4)

def test_model_forward_pass(conv2d_config):
    tf_conv = TensorFlowConvolutions(**conv2d_config)
    model = tf_conv.create_model()
    input_tensor = tf.random.normal(conv2d_config['input_shape'])
    output = model(input_tensor)
    assert output.shape == (conv2d_config['input_shape'][0], conv2d_config['features'][-1])

import jax
import jax.numpy as jnp

def test_compatibility_with_jax():
    key = jax.random.PRNGKey(0)
    jax_array = jax.random.uniform(key, shape=(1, 28, 28, 1), dtype=jnp.float32)
    tf_tensor = convert_array(jax_array, 'tensorflow')
    assert isinstance(tf_tensor, tf.Tensor)
    assert tf_tensor.shape == jax_array.shape

    tf_conv = TensorFlowConvolutions(features=(32, 64), input_shape=(1, 28, 28, 1))
    model = tf_conv.create_model()
    output = model(tf_tensor)
    assert isinstance(output, tf.Tensor)
