import numpy as np
import pennylane as qml

class QuantumCNN:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 3))

    def init(self, key, input_shape):
        self.key = key
        self.input_shape = input_shape
        return self.params

    @qml.qnode(device=qml.device("default.qubit", wires=1))
    def qubit_layer(self, params, input_val):
        qml.RX(input_val, wires=0)
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def quantum_conv_layer(self, inputs, params):
        outputs = []
        for i in range(len(inputs) - 1):
            qml.RX(inputs[i], wires=0)
            qml.RY(inputs[i+1], wires=1)
            qml.CNOT(wires=[0, 1])
            outputs.append(self.qubit_layer(params, inputs[i]))
        return np.array(outputs)

    def forward(self, inputs):
        x = inputs
        for layer in range(self.num_layers):
            x = self.quantum_conv_layer(x, self.params[layer])
        return x

    def apply(self, params, inputs):
        self.params = params
        return self.forward(inputs)

    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        return np.mean((predictions - targets) ** 2)

    def train(self, inputs, targets, num_epochs, learning_rate):
        opt = qml.GradientDescentOptimizer(learning_rate)

        for epoch in range(num_epochs):
            self.params = opt.step(lambda p: self.loss(inputs, targets), self.params)

            if epoch % 10 == 0:
                loss = self.loss(inputs, targets)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, inputs):
        return self.forward(inputs)

def run_qcnn_example():
    # Generate some example data
    num_samples = 100
    input_size = 8
    X = np.random.rand(num_samples, input_size)
    y = np.sin(X.sum(axis=1))  # Simple target function

    # Initialize and train the Quantum CNN
    qcnn = QuantumCNN(num_qubits=2, num_layers=3)
    qcnn.train(X, y, num_epochs=100, learning_rate=0.01)

    # Make predictions
    test_input = np.random.rand(1, input_size)
    prediction = qcnn.predict(test_input)
    print(f"Test input: {test_input}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    run_qcnn_example()
