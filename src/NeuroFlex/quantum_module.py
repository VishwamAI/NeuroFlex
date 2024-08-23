import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
import pennylane as qml

class QuantumModel(nn.Module):
    num_qubits: int = 4
    num_layers: int = 2

    def setup(self):
        self.device = qml.device("default.qubit", wires=self.num_qubits)
        self.qlayer = qml.QNode(self.quantum_circuit, self.device, interface="jax")

    def quantum_circuit(self, inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
        for l in range(self.num_layers):
            qml.StronglyEntanglingLayers(weights[l], wires=range(self.num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    def __call__(self, x):
        weights = self.param('weights', nn.initializers.uniform(scale=0.1), (self.num_layers, self.num_qubits, 3))
        return jax.vmap(self.qlayer, in_axes=(0, None))(x, weights)

    def get_params(self):
        return {'num_qubits': self.num_qubits, 'num_layers': self.num_layers}

def create_quantum_model(num_qubits: int = 4, num_layers: int = 2) -> QuantumModel:
    return QuantumModel(num_qubits=num_qubits, num_layers=num_layers)
