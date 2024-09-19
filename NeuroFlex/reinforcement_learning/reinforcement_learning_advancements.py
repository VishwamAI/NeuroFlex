import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import gym
import logging
import time
from ..utils import utils
from ..core_neural_networks import PyTorchModel, CNN, LSTMModule, LRNN, MachineLearning
from .rl_module import PrioritizedReplayBuffer, RLAgent, RLEnvironment, train_rl_agent

class AdvancedRLAgent(RLAgent):
    def __init__(self, observation_dim: int, action_dim: int, features: List[int] = [64, 64],
                 learning_rate: float = 1e-4, gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01, epsilon_decay: float = 0.995,
                 performance_threshold: float = 0.8, update_interval: int = 86400,
                 buffer_size: int = 100000, batch_size: int = 32):
        super(AdvancedRLAgent, self).__init__(observation_dim, action_dim)
        self.features = features
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.performance_threshold = performance_threshold
        self.update_interval = update_interval
        self.batch_size = batch_size

        self.q_network = self._build_network()
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, (observation_dim,), (action_dim,), batch_size)
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def _build_network(self):
        layers = []
        input_dim = self.observation_dim
        for feature in self.features:
            layers.append(nn.Linear(input_dim, feature))
            layers.append(nn.ReLU())
            input_dim = feature
        layers.append(nn.Linear(input_dim, self.action_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.q_network(x)

    def select_action(self, state: torch.Tensor, training: bool = False) -> int:
        if training and torch.rand(1).item() < self.epsilon:
            return torch.randint(0, self.action_dim, (1,)).item()
        else:
            with torch.no_grad():
                q_values = self(state.unsqueeze(0).to(self.device))
            return q_values.argmax().item()

    def update(self, batch: Dict[str, torch.Tensor]) -> float:
        states = batch['observations'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_observations'].to(self.device)
        dones = batch['dones'].to(self.device)

        # Compute Q-values for current states and actions
        q_values = self(states).gather(1, actions)

        # Compute next state Q-values and select the best actions
        with torch.no_grad():
            next_q_values = self(next_states).max(1, keepdim=True)[0]

        # Compute targets
        targets = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones.unsqueeze(1))

        # Compute loss
        loss = nn.functional.smooth_l1_loss(q_values, targets)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []
        moving_avg_reward = 0
        best_performance = float('-inf')
        window_size = 100  # Size of the moving average window
        no_improvement_count = 0
        max_no_improvement = 50  # Maximum number of episodes without improvement

        for episode in range(num_episodes):
            state, _ = env.reset()
            if state is None:
                state = env.observation_space.sample()
            episode_reward = 0

            for step in range(max_steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).to(self.device)
                action = self.select_action(state_tensor, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    batch = self.replay_buffer.sample()
                    loss = self.update(batch)

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            # Update moving average
            if episode < window_size:
                moving_avg_reward = sum(episode_rewards) / (episode + 1)
            else:
                moving_avg_reward = sum(episode_rewards[-window_size:]) / window_size

            if moving_avg_reward > best_performance:
                best_performance = moving_avg_reward
                no_improvement_count = 0
            else:
                no_improvement_count += 1

            if (episode + 1) % 10 == 0:
                logging.info(f"Episode {episode + 1}, Avg Reward: {moving_avg_reward:.2f}, Epsilon: {self.epsilon:.2f}")

            # Check for early stopping based on performance threshold and improvement
            if moving_avg_reward >= self.performance_threshold:
                if no_improvement_count >= max_no_improvement:
                    logging.info(f"Performance threshold reached and no improvement for {max_no_improvement} episodes. Stopping at episode {episode + 1}")
                    break
            elif no_improvement_count >= max_no_improvement * 2:
                logging.info(f"No significant improvement for {max_no_improvement * 2} episodes. Stopping at episode {episode + 1}")
                break

        self.is_trained = True
        self.performance = moving_avg_reward  # Use the final moving average as performance
        self.last_update = time.time()

        return {"final_reward": self.performance, "episode_rewards": episode_rewards}

    def diagnose(self) -> List[str]:
        issues = []
        if not self.is_trained:
            issues.append("Model is not trained")
        if self.performance < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        return issues

    def heal(self, env, num_episodes: int, max_steps: int, max_attempts: int = 5):
        issues = self.diagnose()
        if issues:
            logging.info(f"Healing issues: {issues}")
            initial_performance = self.performance
            for attempt in range(max_attempts):
                training_info = self.train(env, num_episodes, max_steps)
                new_performance = training_info['final_reward']
                if new_performance > self.performance:
                    self.performance = new_performance
                    self.last_update = time.time()
                    logging.info(f"Healing successful after {attempt + 1} attempts. New performance: {self.performance}")
                    return
                logging.info(f"Healing attempt {attempt + 1} failed. Current performance: {new_performance}")
            logging.warning(f"Failed to improve performance after {max_attempts} attempts. Best performance: {self.performance}")

    def update_model(self, env, num_episodes: int, max_steps: int):
        num_episodes = max(1, num_episodes)
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info['final_reward']
        self.last_update = time.time()

class MultiAgentEnvironment:
    def __init__(self, num_agents: int, env_id: str):
        self.num_agents = num_agents
        self.envs = [gym.make(env_id) for _ in range(num_agents)]

    def reset(self):
        return [torch.tensor(env.reset()[0], dtype=torch.float32) for env in self.envs]

    def step(self, actions):
        def process_action(action):
            if isinstance(action, torch.Tensor):
                return action.item() if action.numel() == 1 else action.tolist()
            return action

        results = [env.step(process_action(action)) for env, action in zip(self.envs, actions)]
        obs, rewards, dones, truncated, infos = zip(*results)
        return (
            torch.stack([torch.tensor(o, dtype=torch.float32) for o in obs]),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.bool),
            torch.tensor(truncated, dtype=torch.bool),
            infos
        )

def create_ppo_agent(env):
    return AdvancedRLAgent(env.observation_space.shape[0], env.action_space.n)

def create_sac_agent(env):
    return AdvancedRLAgent(env.observation_space.shape[0], env.action_space.n)

def train_multi_agent_rl(env: MultiAgentEnvironment, agents: List[AdvancedRLAgent], total_timesteps: int):
    # Initialize episode_rewards and performance history for each agent
    for agent in agents:
        agent.episode_rewards = []
        agent.performance_history = []

    for step in range(total_timesteps):
        observations = env.reset()
        actions = []
        for agent, obs in zip(agents, observations):
            obs_tensor = obs.to(agent.device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32).to(agent.device)
            actions.append(agent.select_action(obs_tensor, training=True))
        next_observations, rewards, dones, truncated, infos = env.step(actions)

        all_agents_trained = True
        for i, (agent, obs, action, next_obs, reward, done) in enumerate(zip(agents, observations, actions, next_observations, rewards, dones)):
            obs_tensor = obs.to(agent.device) if isinstance(obs, torch.Tensor) else torch.tensor(obs, dtype=torch.float32).to(agent.device)
            next_obs_tensor = next_obs.to(agent.device) if isinstance(next_obs, torch.Tensor) else torch.tensor(next_obs, dtype=torch.float32).to(agent.device)
            agent.replay_buffer.add(obs_tensor.cpu().numpy(), action, reward, next_obs_tensor.cpu().numpy(), done)

            # Update the agent if the replay buffer has enough samples
            if len(agent.replay_buffer) > agent.replay_buffer.batch_size:
                batch = agent.replay_buffer.sample(agent.replay_buffer.batch_size)
                loss = agent.update(batch)
                logging.debug(f"Agent {i} update - Step: {step}, Loss: {loss}")

            # Update performance using moving average
            agent.episode_rewards.append(reward)
            if len(agent.episode_rewards) > 100:
                agent.episode_rewards.pop(0)
            agent.performance = sum(agent.episode_rewards) / max(len(agent.episode_rewards), 1)
            agent.performance_history.append(agent.performance)

            # Check if the agent has reached the performance threshold
            if agent.performance < agent.performance_threshold:
                all_agents_trained = False

            # Log agent performance
            if step % 100 == 0:
                logging.info(f"Agent {i} - Step: {step}, Performance: {agent.performance:.4f}, Epsilon: {agent.epsilon:.4f}")

        # Break the loop if all agents have reached the performance threshold
        if all_agents_trained:
            logging.info(f"All agents reached performance threshold at step {step}")
            break

        if all(dones) or all(truncated):
            break

    # Final performance report
    for i, agent in enumerate(agents):
        logging.info(f"Agent {i} final performance: {agent.performance:.4f}")
        agent.plot_performance_history()  # Assuming we add this method to visualize performance over time

    return agents

def advanced_rl_training(env_id: str, num_agents: int, algorithm: str = "PPO", total_timesteps: int = 100000):
    env = MultiAgentEnvironment(num_agents, env_id)

    if algorithm in ["PPO", "SAC"]:
        agents = [create_ppo_agent(gym.make(env_id)) for _ in range(num_agents)]
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    trained_agents = train_multi_agent_rl(env, agents, total_timesteps)
    return trained_agents

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    trained_agents = advanced_rl_training("CartPole-v1", num_agents=3, algorithm="PPO", total_timesteps=50000)
    logging.info(f"Trained {len(trained_agents)} agents using PPO")

    # Demonstrate self-healing
    for i, agent in enumerate(trained_agents):
        issues = agent.diagnose()
        if issues:
            logging.info(f"Agent {i} detected issues: {issues}")
            agent.heal(gym.make("CartPole-v1"), num_episodes=100, max_steps=500)
            logging.info(f"Agent {i} healed. New performance: {agent.performance}")
