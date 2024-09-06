import pytest
import tensorflow as tf
import jax
import jax.numpy as jnp
from NeuroFlex.core_neural_networks.tensorflow.tensorflow_module import TensorFlowModel, create_tensorflow_model, train_tensorflow_model, tensorflow_predict
from NeuroFlex.core_neural_networks.tensorflow.tensorflow_convolutions import TensorFlowConvolutions, create_conv_model, train_conv_model, conv_predict
from NeuroFlex.utils.utils import convert_array

@pytest.fixture
def conv2d_config():
    return {
        'input_shape': (28, 28, 1),
        'output_dim': 10,
        'hidden_layers': [32, 64, 128]
    }

@pytest.fixture
def conv3d_config():
    return {
        'input_shape': (16, 16, 16, 1),
        'output_dim': 10,
        'hidden_layers': [16, 32, 64]
    }

def test_tensorflow_model_init(conv2d_config):
    tf_model = create_tensorflow_model(**conv2d_config)
    assert isinstance(tf_model, TensorFlowModel)
    assert len(tf_model.hidden_layers) == len(conv2d_config['hidden_layers'])
    assert all(isinstance(layer, tf.keras.layers.Dense) for layer in tf_model.hidden_layers)

def test_model_call(conv2d_config):
    tf_model = create_tensorflow_model(**conv2d_config)
    input_tensor = tf.random.normal((1,) + conv2d_config['input_shape'])
    output = tf_model(input_tensor)
    assert isinstance(output, tf.Tensor)
    assert output.shape == (1, conv2d_config['output_dim'])

def test_create_tensorflow_model(conv2d_config):
    model = create_tensorflow_model(**conv2d_config)
    assert isinstance(model, TensorFlowModel)
    assert model.input_shape == (None,) + conv2d_config['input_shape']
    assert model.output_layer.units == conv2d_config['output_dim']

def test_train_tensorflow_model(conv2d_config):
    model = create_tensorflow_model(**conv2d_config)
    x_train = tf.random.normal((100,) + conv2d_config['input_shape'])
    y_train = tf.random.normal((100, conv2d_config['output_dim']))
    history = train_tensorflow_model(model, x_train, y_train, epochs=1)
    assert isinstance(history, tf.keras.callbacks.History)

def test_tensorflow_predict(conv2d_config):
    model = create_tensorflow_model(**conv2d_config)
    x = tf.random.normal((1,) + conv2d_config['input_shape'])
    output = tensorflow_predict(model, x)
    assert isinstance(output, tf.Tensor)
    assert output.shape == (1, conv2d_config['output_dim'])

def test_compatibility_with_jax():
    key = jax.random.PRNGKey(0)
    jax_array = jax.random.uniform(key, shape=(1, 28, 28, 1), dtype=jnp.float32)
    tf_tensor = convert_array(jax_array, 'tensorflow')
    assert isinstance(tf_tensor, tf.Tensor)
    assert tf_tensor.shape == jax_array.shape

    conv_config = {
        'input_shape': (28, 28, 1),
        'output_dim': 10,
        'hidden_layers': [32, 64]
    }
    model = create_tensorflow_model(**conv_config)
    output = tensorflow_predict(model, tf_tensor)
    assert isinstance(output, tf.Tensor)
    assert output.shape == (1, conv_config['output_dim'])

def test_tensorflow_convolutions():
    input_shape = (28, 28, 1)
    filters = [32, 64]
    kernel_sizes = [(3, 3), (3, 3)]
    strides = [(1, 1), (1, 1)]
    paddings = ['same', 'same']

    conv_layers = TensorFlowConvolutions(filters, kernel_sizes, strides, paddings)
    assert len(conv_layers.conv_layers) == len(filters)

    input_tensor = tf.random.normal((1,) + input_shape)
    output = conv_layers(input_tensor)
    assert isinstance(output, tf.Tensor)
    assert len(output.shape) == 2  # Flattened output

def test_create_conv_model():
    input_shape = (28, 28, 1)
    filters = [32, 64]
    kernel_sizes = [(3, 3), (3, 3)]
    strides = [(1, 1), (1, 1)]
    paddings = ['same', 'same']
    output_dim = 10

    model = create_conv_model(input_shape, filters, kernel_sizes, strides, paddings, output_dim)
    assert isinstance(model, tf.keras.Model)

    input_tensor = tf.random.normal((1,) + input_shape)
    output = model(input_tensor)
    assert isinstance(output, tf.Tensor)
    assert output.shape == (1, output_dim)

def test_train_conv_model():
    input_shape = (28, 28, 1)
    filters = [32, 64]
    kernel_sizes = [(3, 3), (3, 3)]
    strides = [(1, 1), (1, 1)]
    paddings = ['same', 'same']
    output_dim = 10

    model = create_conv_model(input_shape, filters, kernel_sizes, strides, paddings, output_dim)
    x_train = tf.random.normal((100,) + input_shape)
    y_train = tf.random.normal((100, output_dim))

    history = train_conv_model(model, x_train, y_train, epochs=1)
    assert isinstance(history, tf.keras.callbacks.History)

def test_conv_predict():
    input_shape = (28, 28, 1)
    filters = [32, 64]
    kernel_sizes = [(3, 3), (3, 3)]
    strides = [(1, 1), (1, 1)]
    paddings = ['same', 'same']
    output_dim = 10

    model = create_conv_model(input_shape, filters, kernel_sizes, strides, paddings, output_dim)
    x = tf.random.normal((1,) + input_shape)
    output = conv_predict(model, x)
    assert isinstance(output, tf.Tensor)
    assert output.shape == (1, output_dim)
