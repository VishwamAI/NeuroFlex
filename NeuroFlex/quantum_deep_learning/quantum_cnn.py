import numpy as np
import pennylane as qml
import jax
import jax.numpy as jnp

class QuantumCNN:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 3))
        self.qubit_layer = qml.qnode(device=qml.device("default.qubit", wires=1))(self._qubit_layer)

    def init(self, key, input_shape):
        self.key = key
        self.input_shape = input_shape
        return self.params

    def _qubit_layer(self, params, input_val):
        qml.RX(input_val, wires=0)
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

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
        expected_output_size = max(1, inputs.shape[0] - self.num_layers)
        for layer in range(self.num_layers):
            x = self.quantum_conv_layer(x, self.params[layer])
            print(f"Layer {layer} output shape: {x.shape}")
            # Ensure x is 2D and reduce input size only if necessary
            if layer < self.num_layers - 1:
                x = x[:-1]  # Remove only one element per layer
                print(f"After reduction, shape: {x.shape}")
            else:
                print(f"No reduction needed at layer {layer}")
            if x.shape[0] <= 1:
                print(f"Cannot reduce further, breaking at layer {layer}")
                break  # Stop if we can't reduce further
        # Reshape while maintaining the total number of elements
        final_output = x.reshape(-1, x.shape[-1])
        if final_output.shape[0] > expected_output_size:
            final_output = final_output[:expected_output_size]
        elif final_output.shape[0] < expected_output_size:
            padding = jnp.zeros((expected_output_size - final_output.shape[0], final_output.shape[1]))
            final_output = jnp.vstack((final_output, padding))
        if final_output.shape[1] > 1:
            final_output = final_output.mean(axis=1, keepdims=True)
        else:
            final_output = final_output.reshape(expected_output_size, 1)
        print(f"Final output shape: {final_output.shape}")
        print(f"Expected output shape: ({expected_output_size}, 1)")
        return final_output

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        loss_fn = lambda params: self.loss(params, inputs, targets)
        grads = jax.grad(loss_fn)(self.params)
        print(f"Gradients: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
        new_params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)
        print(f"Param diff: {jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), self.params, new_params)}")
        self.params = new_params

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        loss_fn = lambda params: self.loss(params, inputs, targets)
        grads = jax.grad(loss_fn)(self.params)
        print(f"Gradients: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
        print(f"Gradient details: {jax.tree_map(lambda x: x, grads)}")
        new_params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)
        print(f"Param diff: {jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), self.params, new_params)}")
        print(f"New params: {jax.tree_map(lambda x: x, new_params)}")
        self.params = new_params

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        loss_fn = lambda params: self.loss(params, inputs, targets)
        grads = jax.grad(loss_fn)(self.params)
        print(f"Gradients: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
        print(f"Gradient details: {jax.tree_map(lambda x: x, grads)}")
        new_params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)
        print(f"Param diff: {jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), self.params, new_params)}")
        print(f"New params: {jax.tree_map(lambda x: x, new_params)}")
        self.params = new_params

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        loss_fn = lambda params: self.loss(params, inputs, targets)
        grads = jax.grad(loss_fn)(self.params)
        print(f"Gradients: {jax.tree_map(lambda x: jnp.sum(jnp.abs(x)), grads)}")
        new_params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)
        print(f"Param diff: {jax.tree_map(lambda x, y: jnp.sum(jnp.abs(x - y)), self.params, new_params)}")
        self.params = new_params

    def gradient_step(self, inputs, targets, learning_rate):
        print(f"gradient_step input shape: {inputs.shape}")
        loss_fn = lambda params: self.loss(params, inputs, targets)
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

    def gradient_step(self, inputs, targets, learning_rate):
        grads = jax.grad(self.loss)(self.params, inputs, targets)
        self.params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)

    def apply(self, params, inputs):
        self.params = params
        return self.forward(inputs)

    def loss(self, inputs, targets):
        predictions = self.forward(inputs)
        return jnp.mean((predictions - targets) ** 2)

    def gradient_step(self, inputs, targets, learning_rate):
        grads = jax.grad(self.loss)(self.params, inputs, targets)
        self.params = jax.tree_map(lambda p, g: p - learning_rate * g, self.params, grads)

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
