import numpy as np
import pennylane as qml

class QuantumReinforcementLearning:
    def __init__(self, num_qubits, num_actions):
        self.num_qubits = num_qubits
        self.num_actions = num_actions
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_qubits, 3))

    @qml.qnode(device=qml.device("default.qubit", wires=1))
    def qubit_layer(self, params, state):
        qml.RX(state, wires=0)
        qml.RY(params[0], wires=0)
        qml.RZ(params[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    def quantum_circuit(self, state):
        outputs = []
        for i in range(self.num_qubits):
            outputs.append(self.qubit_layer(self.params[i], state[i]))
        return np.array(outputs)

    def get_action(self, state):
        q_values = self.quantum_circuit(state)
        return np.argmax(q_values)

    def update(self, state, action, reward, next_state, learning_rate):
        current_q = self.quantum_circuit(state)[action]
        next_q = np.max(self.quantum_circuit(next_state))
        target = reward + 0.99 * next_q  # 0.99 is the discount factor

        loss = (target - current_q) ** 2
        grad = qml.grad(self.quantum_circuit)(state)[action]
        self.params[action] -= learning_rate * loss * grad

    def train(self, env, num_episodes, max_steps, learning_rate):
        for episode in range(num_episodes):
            state = env.reset()
            total_reward = 0

            for step in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                self.update(state, action, reward, next_state, learning_rate)

                state = next_state
                total_reward += reward

                if done:
                    break

            if episode % 10 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

    def evaluate(self, env, num_episodes, max_steps):
        total_rewards = []
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0

            for _ in range(max_steps):
                action = self.get_action(state)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state

                if done:
                    break

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average Reward over {num_episodes} episodes: {avg_reward}")
        return avg_reward

def run_qrl_example():
    import gym

    # Create a simple environment
    env = gym.make('CartPole-v1')

    # Initialize the Quantum Reinforcement Learning agent
    qrl = QuantumReinforcementLearning(num_qubits=4, num_actions=2)

    # Train the agent
    qrl.train(env, num_episodes=100, max_steps=200, learning_rate=0.01)

    # Evaluate the agent
    qrl.evaluate(env, num_episodes=10, max_steps=200)

if __name__ == "__main__":
    run_qrl_example()
