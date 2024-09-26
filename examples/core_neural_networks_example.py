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

"""
NeuroFlex Core Neural Networks Example

This script demonstrates the usage of core neural networks in NeuroFlex.
It includes initialization, configuration, and basic operations.
"""

import numpy as np
from NeuroFlex.core_neural_networks import NeuralNetwork, Layer, Activation
from NeuroFlex.utils import data_loader

def main():
    # Initialize a neural network
    nn = NeuralNetwork()

    # Add layers to the neural network
    nn.add(Layer(input_size=10, output_size=64))
    nn.add(Activation('relu'))
    nn.add(Layer(input_size=64, output_size=32))
    nn.add(Activation('relu'))
    nn.add(Layer(input_size=32, output_size=1))
    nn.add(Activation('sigmoid'))

    # Configure the neural network
    nn.compile(loss='binary_crossentropy', optimizer='adam')

    # Load example data
    X_train, y_train = data_loader.load_example_data()

    # Train the neural network
    nn.fit(X_train, y_train, epochs=10, batch_size=32)

    # Make predictions
    X_test = np.random.rand(5, 10)  # 5 samples, 10 features each
    predictions = nn.predict(X_test)

    print("Predictions:")
    print(predictions)

if __name__ == "__main__":
    main()
