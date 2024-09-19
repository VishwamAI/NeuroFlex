# This __init__.py file marks the directory as a Python module.
# It allows Python to recognize this directory as a package and enables importing from it.

from .tensorflow_module import TensorFlowModel, create_tensorflow_model, train_tensorflow_model, tensorflow_predict
from .tensorflow_convolutions import TensorFlowConvolutions, create_conv_model, train_conv_model, conv_predict

__all__ = [
    'TensorFlowModel', 'create_tensorflow_model', 'train_tensorflow_model', 'tensorflow_predict',
    'TensorFlowConvolutions', 'create_conv_model', 'train_conv_model', 'conv_predict'
]

# Add any TensorFlow specific functions or variables here
def get_tensorflow_version():
    import tensorflow as tf
    return tf.__version__

# You can add more TensorFlow specific functions or constants as needed
SUPPORTED_TF_LAYERS = ['Dense', 'Conv2D', 'LSTM', 'GRU']

# Example of a utility function that could be useful for TensorFlow models
def create_mlp(input_shape, hidden_layers, num_classes):
    import tensorflow as tf
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation='relu'))
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
    return model

# Add any initialization code specific to TensorFlow here
def initialize_tensorflow():
    print("Initializing TensorFlow Module...")
    # Add any necessary initialization code here
