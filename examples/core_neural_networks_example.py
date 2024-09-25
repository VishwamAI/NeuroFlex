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
