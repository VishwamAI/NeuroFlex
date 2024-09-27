import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile, execute
from qiskit.providers.aer import Aer
from qiskit.circuit.library import ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA
from qiskit.circuit import Parameter
from qiskit.quantum_info import random_statevector

class QuantumPredictiveModel:
    def __init__(self, num_qubits):
        self.num_qubits = num_qubits
        self.feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2)
        self.variational_form = self._create_variational_form()
        self.optimizer = COBYLA(maxiter=50)  # Reduced from 100 to 50
        self.optimal_params = None  # Initialize optimal_params

    def _create_variational_form(self):
        qc = QuantumCircuit(self.num_qubits)
        params = [Parameter(f'θ_{i}') for i in range(self.num_qubits * 2)]
        for i in range(self.num_qubits):
            qc.rx(params[i*2], i)
            qc.ry(params[i*2+1], i)
        if self.num_qubits > 1:
            qc.cx(0, 1)
        return qc

    def train(self, X, y):
        def objective_function(params):
            print(f"Objective function called with params: {params}")
            qc = QuantumCircuit(self.num_qubits, 1)
            qc.append(self.feature_map, range(self.num_qubits))
            qc = qc.compose(self.variational_form)

            # Ensure all parameters are accounted for
            all_params = set(self.feature_map.parameters) | set(self.variational_form.parameters)
            param_dict = dict(zip(all_params, params))

            print("Binding parameters to circuit")
            bound_circuit = qc.bind_parameters(param_dict)
            bound_circuit.measure(self.num_qubits-1, 0)

            print("Getting backend")
            backend = Aer.get_backend('aer_simulator')
            print("Transpiling circuit")
            transpiled_circuit = transpile(bound_circuit, backend)

            print("Running job")
            job = backend.run(transpiled_circuit, shots=500)
            print("Getting results")
            result = job.result().get_counts(transpiled_circuit)
            prediction = result.get('1', 0) / 500

            loss = np.mean((y - prediction)**2)
            print(f"Loss: {loss}")
            return loss

        print("Starting optimization")
        initial_params = np.random.rand(len(self.feature_map.parameters) + len(self.variational_form.parameters))
        result = self.optimizer.minimize(objective_function, x0=initial_params)
        self.optimal_params = result.x
        self.is_trained = True
        print("Optimization completed")

    def predict(self, X):
        if self.optimal_params is None:
            raise ValueError("Model has not been trained. Call train() before predict().")
        qc = QuantumCircuit(self.num_qubits, 1)
        qc.append(self.feature_map, range(self.num_qubits))
        qc = qc.compose(self.variational_form)

        all_params = list(self.feature_map.parameters) + list(self.variational_form.parameters)
        param_dict = dict(zip(all_params, self.optimal_params))

        bound_circuit = qc.bind_parameters(param_dict)
        bound_circuit.measure(self.num_qubits-1, 0)

        backend = Aer.get_backend('aer_simulator')
        transpiled_circuit = transpile(bound_circuit, backend)
        job = backend.run(transpiled_circuit, shots=500)
        result = job.result().get_counts(bound_circuit)
        return result.get('1', 0) / 500

class QuantumEncryption:
    def __init__(self, key_size):
        self.key_size = key_size

    def generate_key(self):
        qc = QuantumCircuit(self.key_size, self.key_size)
        for i in range(self.key_size):
            qc.h(i)
        qc.measure(range(self.key_size), range(self.key_size))

        job = execute(qc, Aer.get_backend('qasm_simulator'), shots=1)
        result = job.result().get_counts(qc)
        key = list(result.keys())[0]
        return [int(bit) for bit in key]

    def encrypt(self, message, key):
        return [m ^ k for m, k in zip(message, key)]

    def decrypt(self, ciphertext, key):
        return [c ^ k for c, k in zip(ciphertext, key)]

# Example usage
if __name__ == "__main__":
    # Quantum Predictive Model
    qpm = QuantumPredictiveModel(num_qubits=4)
    X_train = np.random.rand(100, 4)
    y_train = np.random.randint(2, size=100)
    qpm.train(X_train, y_train)

    X_test = np.random.rand(10, 4)
    predictions = [qpm.predict(x) for x in X_test]
    print("Quantum Predictive Model Predictions:", predictions)

    # Quantum Encryption
    qe = QuantumEncryption(key_size=8)
    key = qe.generate_key()
    message = [1, 0, 1, 1, 0, 0, 1, 0]
    ciphertext = qe.encrypt(message, key)
    decrypted = qe.decrypt(ciphertext, key)

    print("Original message:", message)
    print("Encrypted message:", ciphertext)
    print("Decrypted message:", decrypted)
