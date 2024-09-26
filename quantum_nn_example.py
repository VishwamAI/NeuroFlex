import pennylane as qml
import numpy as np
from pennylane import numpy as pnp
import matplotlib.pyplot as plt

# Set up the quantum device
dev = qml.device("default.qubit", wires=2)

# Define the quantum circuit
@qml.qnode(dev)
def quantum_circuit(inputs, weights):
    # Encode the classical input data
    qml.RY(inputs[0], wires=0)
    qml.RY(inputs[1], wires=1)

    # Apply parameterized quantum gates
    qml.RZ(weights[0], wires=0)
    qml.RX(weights[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[2], wires=1)

    # Measure the output
    return qml.expval(qml.PauliZ(1))

# Define a hybrid quantum-classical model
def hybrid_model(inputs, weights):
    q_out = quantum_circuit(inputs, weights)
    return pnp.tanh(q_out)  # Apply classical non-linearity

# Generate some dummy data
X = pnp.array(np.random.rand(100, 2), dtype=float)
y = pnp.array(np.sum(X, axis=1) > 1, dtype=float)

# Initialize weights
weights = pnp.random.uniform(low=0, high=2*np.pi, size=3)

# Define the cost function
def cost(weights, X, y):
    predictions = pnp.array([hybrid_model(x, weights) for x in X])
    return pnp.mean((predictions - y) ** 2)

# Optimize the model
opt = qml.GradientDescentOptimizer(stepsize=0.1)
steps = 100
cost_history = []

for i in range(steps):
    weights = opt.step(lambda w: cost(w, X, y), weights)
    cost_value = cost(weights, X, y)
    cost_history.append(cost_value)
    if (i + 1) % 10 == 0:
        print(f"Step {i+1}, Cost: {cost_value:.4f}")

# Plot the training progress
plt.plot(range(1, steps + 1), cost_history)
plt.xlabel('Optimization step')
plt.ylabel('Cost')
plt.title('Training Progress')
plt.show()

# Test the trained model
test_input = np.array([0.6, 0.7])
prediction = hybrid_model(test_input, weights)
print(f"Prediction for input {test_input}: {prediction:.4f}")

# Visualize the decision boundary
x1 = np.linspace(0, 1, 20)
x2 = np.linspace(0, 1, 20)
X1, X2 = np.meshgrid(x1, x2)
Z = np.array([[hybrid_model(np.array([x1, x2]), weights) for x1, x2 in zip(X1_row, X2_row)] for X1_row, X2_row in zip(X1, X2)])

plt.contourf(X1, X2, Z, levels=20, cmap='RdBu', alpha=0.8)
plt.colorbar(label='Model Output')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.xlabel('Input 1')
plt.ylabel('Input 2')
plt.title('Decision Boundary of Hybrid Quantum-Classical Model')
plt.show()

print("Implementation complete. The example demonstrates a hybrid quantum-classical model using PennyLane.")
