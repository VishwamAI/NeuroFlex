import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp

@qml.qnode(device=qml.device("default.qubit", wires=1), interface="jax")
def qubit_layer(params, input_val):
    qml.RX(input_val, wires=0)
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))  # Return scalar value directly

class QuantumCNN:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = None

    def init(self, key, input_shape):
        self.key = key
        self.input_shape = input_shape
        # Initialize parameters randomly
        self.params = jax.random.normal(key, (self.num_layers, self.num_qubits, 2))
        return self.params

    def apply(self, params, x):
        self.params = params
        return self.forward(x)

    def quantum_conv_layer(self, params, x):
        print(f"quantum_conv_layer input shape: {x.shape}, values: {x}")
        print(f"quantum_conv_layer params shape: {params.shape}, values: {params}")

        if x.shape[0] == 1:
            # If input size is 1, return the input as is
            output = []
            for j in range(self.num_qubits):
                input_val = float(x[0, j])
                qubit_output = qubit_layer(params[j], input_val=input_val)
                output.append(qubit_output)
            output_array = jnp.array([output])
        else:
            output = []
            for i in range(x.shape[0] - 1):  # Reduce output size by 1
                layer_output = []
                for j in range(self.num_qubits):
                    input_val = float((x[i, j] + x[i+1, j]) / 2)  # Ensure input_val is a scalar
                    print(f"input_val for qubit {j}: {input_val}")
                    qubit_output = qubit_layer(params[j], input_val=input_val)  # Average of two adjacent inputs
                    print(f"qubit_output for qubit {j}: {qubit_output}")
                    layer_output.append(qubit_output)
                output.append(layer_output)
            output_array = jnp.array(output)

        print(f"quantum_conv_layer output shape: {output_array.shape}, values: {output_array}")
        return output_array

    def forward(self, x):
        for layer in range(self.num_layers):
            x = self.quantum_conv_layer(self.params[layer], x)
        return x

@qml.qnode(qml.device('default.qubit', wires=1))
def qubit_layer(params, input_val):
    qml.RX(input_val, wires=0)
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

def _qubit_layer(self, params, input_val):
    return qubit_layer(params, input_val)

    def quantum_conv_layer(self, inputs, params):
        print(f"quantum_conv_layer input shape: {inputs.shape}")
        print(f"quantum_conv_layer params shape: {params.shape}")
        def apply_layer(i, x):
            qml.RX(x, wires=0)
            qml.RY(inputs[i+1], wires=1)
            qml.CNOT(wires=[0, 1])
            param_index = jax.lax.rem(i, params.shape[0])
            slice_params = jax.lax.dynamic_slice(params, (param_index, 0), (1, params.shape[1])).squeeze()
            print(f"apply_layer i={i}, x={x}, param_index={param_index}, slice_params={slice_params}")
            return self.qubit_layer(params=slice_params, input_val=x)

        if inputs.shape[0] > 1:
            outputs = jax.vmap(apply_layer, in_axes=(0, 0))(jnp.arange(inputs.shape[0] - 1), inputs[:-1])
            print(f"quantum_conv_layer output shape: {outputs.shape}")
            reshaped_outputs = outputs.reshape(-1, self.num_qubits)
            print(f"quantum_conv_layer reshaped output shape: {reshaped_outputs.shape}")
            return reshaped_outputs  # Reshape to ensure consistent output shape
        else:
            print("quantum_conv_layer: input shape <= 1, returning zeros")
            return jnp.zeros((1, self.num_qubits))  # Return a 2D array with one row and correct number of columns

    def forward(self, inputs):
        x = inputs
        print(f"forward input shape: {x.shape}")
        for layer in range(self.num_layers):
            x = self.quantum_conv_layer(x, self.params[layer])
            print(f"Layer {layer} output shape: {x.shape}")
            if x.shape[0] <= 1:
                print(f"Cannot reduce further, breaking at layer {layer}")
                break  # Stop if we can't reduce further

        # Ensure the output is 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)

        # If the output has more than one column, take the mean across columns
        if x.shape[1] > 1:
            x = x.mean(axis=1, keepdims=True)

        print(f"Final output shape: {x.shape}")
        return x

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        def loss_fn(params):
            predictions = self.apply(params, inputs)
            return jnp.mean(jnp.square(predictions - targets))

        grads = jax.grad(loss_fn)(self.params)
        print(f"Gradients: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
        print(f"Gradient details: {jax.tree_map(lambda x: x, grads)}")

        new_params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)
        print(f"Param diff: {jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), self.params, new_params)}")
        print(f"New params: {jax.tree_map(lambda x: x, new_params)}")

        self.params = new_params

    def calculate_output_shape(self, input_shape):
        output_shape = input_shape[0]
        for _ in range(self.num_layers):
            output_shape -= 1
        return (output_shape, 1)

    def apply(self, params, inputs):
        return self.forward(inputs)

    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        return jnp.mean((predictions - targets) ** 2)

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
