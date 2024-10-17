import numpy as np
import pennylane as qml

class QuantumGenerativeModel:
    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_layers, num_qubits, 3))

    @qml.qnode(device=qml.device("default.qubit", wires=1))
    def qubit_layer(self, params):
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        qml.RZ(params[2], wires=0)
        return qml.expval(qml.PauliZ(0))

    def quantum_circuit(self, noise):
        for layer in range(self.num_layers):
            for qubit in range(self.num_qubits):
                qml.RX(noise[qubit] + self.params[layer, qubit, 0], wires=qubit)
                qml.RY(self.params[layer, qubit, 1], wires=qubit)
                qml.RZ(self.params[layer, qubit, 2], wires=qubit)
            if layer < self.num_layers - 1:
                for qubit in range(self.num_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
        return [qml.expval(qml.PauliZ(i)) for i in range(self.num_qubits)]

    @qml.qnode(device=qml.device("default.qubit", wires=1))
    def discriminator(self, x, params):
        qml.RX(x, wires=0)
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def generate_sample(self, noise):
        return self.quantum_circuit(noise)

    def train(self, real_data, num_epochs, batch_size, learning_rate):
        optimizer = qml.GradientDescentOptimizer(learning_rate)
        disc_params = np.random.uniform(low=-np.pi, high=np.pi, size=2)

        for epoch in range(num_epochs):
            for _ in range(len(real_data) // batch_size):
                # Train Discriminator
                real_batch = np.random.choice(real_data, size=batch_size)
                noise = np.random.normal(0, 1, size=(batch_size, self.num_qubits))
                fake_batch = np.array([self.generate_sample(n) for n in noise])

                disc_real = np.mean([self.discriminator(x, disc_params) for x in real_batch])
                disc_fake = np.mean([self.discriminator(x, disc_params) for x in fake_batch])
                disc_loss = -(np.log(disc_real) + np.log(1 - disc_fake))

                disc_params = optimizer.step(lambda p: self.discriminator(real_batch[0], p), disc_params)

                # Train Generator
                noise = np.random.normal(0, 1, size=(batch_size, self.num_qubits))
                fake_batch = np.array([self.generate_sample(n) for n in noise])
                gen_loss = -np.mean([self.discriminator(x, disc_params) for x in fake_batch])

                self.params = optimizer.step(lambda p: self.quantum_circuit(noise[0]), self.params)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Disc Loss: {disc_loss:.4f}, Gen Loss: {gen_loss:.4f}")

    def generate(self, num_samples):
        noise = np.random.normal(0, 1, size=(num_samples, self.num_qubits))
        return np.array([self.generate_sample(n) for n in noise])

def run_qgm_example():
    # Generate some example data
    num_samples = 1000
    real_data = np.random.normal(0, 1, size=num_samples)

    # Initialize and train the Quantum Generative Model
    qgm = QuantumGenerativeModel(num_qubits=4, num_layers=3)
    qgm.train(real_data, num_epochs=100, batch_size=32, learning_rate=0.01)

    # Generate samples
    generated_samples = qgm.generate(num_samples=1000)

    # Compare distributions
    import matplotlib.pyplot as plt
    plt.hist(real_data, bins=50, alpha=0.5, label='Real Data')
    plt.hist(generated_samples.flatten(), bins=50, alpha=0.5, label='Generated Data')
    plt.legend()
    plt.title('Comparison of Real and Generated Data Distributions')
    plt.savefig('qgm_distribution_comparison.png')
    plt.close()

    print("Distribution comparison plot saved as 'qgm_distribution_comparison.png'")

if __name__ == "__main__":
    run_qgm_example()
