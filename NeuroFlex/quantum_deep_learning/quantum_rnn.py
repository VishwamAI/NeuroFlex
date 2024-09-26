import numpy as np
import pennylane as qml

class QuantumRNN:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 3))

    @qml.qnode(device=qml.device("default.qubit", wires=1))
    def qubit_layer(self, params, input_val):
        qml.RX(input_val, wires=0)
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def quantum_rnn_layer(self, inputs, params, hidden_state):
        outputs = []
        for t in range(len(inputs)):
            qml.RX(hidden_state, wires=0)
            qml.RY(inputs[t], wires=1)
            qml.CNOT(wires=[0, 1])
            hidden_state = self.qubit_layer(params, inputs[t])
            outputs.append(hidden_state)
        return np.array(outputs), hidden_state

    def forward(self, inputs):
        hidden_state = 0
        x = inputs
        for layer in range(self.num_layers):
            x, hidden_state = self.quantum_rnn_layer(x, self.params[layer], hidden_state)
        return x

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

def run_qrnn_example():
    # Generate some example data
    num_samples = 100
    sequence_length = 10
    X = np.random.rand(num_samples, sequence_length)
    y = np.sin(X.cumsum(axis=1))  # Simple target function

    # Initialize and train the Quantum RNN
    qrnn = QuantumRNN(num_qubits=2, num_layers=3)
    qrnn.train(X, y, num_epochs=100, learning_rate=0.01)

    # Make predictions
    test_input = np.random.rand(1, sequence_length)
    prediction = qrnn.predict(test_input)
    print(f"Test input: {test_input}")
    print(f"Prediction: {prediction}")

if __name__ == "__main__":
    run_qrnn_example()
