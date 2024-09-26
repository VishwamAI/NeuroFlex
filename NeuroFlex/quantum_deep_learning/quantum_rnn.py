import numpy as np
import pennylane as qml

class QuantumRNN:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.param_init_fn = lambda key, shape: np.random.uniform(low=-np.pi, high=np.pi, size=shape)
        self.params = self.param_init_fn(None, (num_layers, num_qubits, 2))

    def init(self, key, input_shape):
        self.key = key
        self.input_shape = input_shape
        # Initialize params for 3D input shape (batch_size, time_steps, features)
        self.params = self.param_init_fn(key, (self.num_layers, self.num_qubits, 2))
        return self.params

    def qubit_layer(self, params, input_val):
        dev = qml.device("default.qubit", wires=self.num_qubits)
        @qml.qnode(dev)
        def _qubit_circuit(params, input_val):
            input_val = np.atleast_2d(input_val)
            for b in range(input_val.shape[0]):  # Iterate over batch
                for i in range(self.num_qubits):
                    qml.RX(input_val[b, i % input_val.shape[1]], wires=i)
                    qml.RY(params[i][0], wires=i)
                    qml.RZ(params[i][1], wires=i)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]
        return np.array([_qubit_circuit(params, iv) for iv in input_val])

    def quantum_rnn_layer(self, inputs, params, hidden_state):
        batch_size, time_steps, input_features = inputs.shape
        hidden_features = self.num_qubits  # Number of features in hidden state

        # Initialize hidden_state if it's the first pass
        if isinstance(hidden_state, int) and hidden_state == 0:
            hidden_state = np.zeros((batch_size, hidden_features))

        outputs = np.zeros((batch_size, time_steps, hidden_features))

        for t in range(time_steps):
            input_t = inputs[:, t, :]
            # Ensure hidden_state has the same batch size as input_t
            if hidden_state.shape[0] != batch_size:
                hidden_state = np.broadcast_to(hidden_state, (batch_size, hidden_features))

            # Combine hidden state and input, ensuring dimensions match
            combined_input = np.concatenate([hidden_state, input_t], axis=1)

            # Adjust combined_input if necessary to match num_qubits
            if combined_input.shape[1] > self.num_qubits:
                combined_input = combined_input[:, :self.num_qubits]
            elif combined_input.shape[1] < self.num_qubits:
                pad_width = self.num_qubits - combined_input.shape[1]
                combined_input = np.pad(combined_input, ((0, 0), (0, pad_width)), mode='constant')

            hidden_state = self.qubit_layer(params=params, input_val=combined_input)
            outputs[:, t, :] = hidden_state

        return outputs, hidden_state

    def forward(self, inputs):
        hidden_state = 0
        x = inputs
        for layer in range(self.num_layers):
            x, hidden_state = self.quantum_rnn_layer(x, self.params[layer], hidden_state)
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
