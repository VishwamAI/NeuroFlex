import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any
import logging
import time
from .rl_module import PrioritizedReplayBuffer, RLEnvironment
from ..utils import utils


class SelfCuringRLAgent(nn.Module):
    def __init__(
        self,
        features: List[int],
        action_dim: int,
        learning_rate: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        performance_threshold: float = 0.8,
        update_interval: int = 86400,
    ):  # 24 hours in seconds
        super(SelfCuringRLAgent, self).__init__()
        self.features = features
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.performance_threshold = performance_threshold
        self.update_interval = update_interval

        self.q_network = nn.Sequential(
            *[
                nn.Linear(features[i], features[i + 1])
                for i in range(len(features) - 1)
            ],
            nn.Linear(features[-1], action_dim),
        )
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.replay_buffer = PrioritizedReplayBuffer(
            100000, (features[0],), (action_dim,)
        )
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

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
        # Check batch size consistency
        batch_size = batch["observations"].shape[0]
        for key, tensor in batch.items():
            assert (
                tensor.shape[0] == batch_size
            ), f"Inconsistent batch size for {key}: {tensor.shape[0]} != {batch_size}"

        states = batch["observations"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_observations"].to(self.device)
        dones = batch["dones"].to(self.device)

        logging.debug(
            f"Initial shapes - States: {states.shape}, Actions: {actions.shape}, Rewards: {rewards.shape}, Next States: {next_states.shape}, Dones: {dones.shape}"
        )

        # Ensure actions tensor has the correct shape
        actions = actions.long()[
            :, 0
        ]  # Use only the first column and ensure actions are long integers
        logging.debug(f"Actions shape after adjustment: {actions.shape}")

        # Compute Q-values for current states and actions
        q_values = self(states)
        logging.debug(f"Q-values shape before gather: {q_values.shape}")
        q_values = q_values.gather(1, actions.unsqueeze(-1))
        logging.debug(f"Q-values shape after gather: {q_values.shape}")

        # Additional debug logging
        logging.debug(f"Actions sample: {actions[:5].tolist()}")
        logging.debug(f"Q-values sample after gather: {q_values[:5].tolist()}")

        # Compute next state Q-values and select the best actions
        with torch.no_grad():
            next_q_values = self(next_states).max(1, keepdim=True)[0]
        logging.debug(f"Next Q-values shape: {next_q_values.shape}")

        # Compute targets
        targets = rewards.unsqueeze(1) + self.gamma * next_q_values * (
            ~dones.unsqueeze(1)
        )
        logging.debug(f"Targets shape after computation: {targets.shape}")

        # Detach targets from computation graph to prevent gradients from flowing through them
        targets = targets.detach()

        # Debug information
        logging.debug(
            f"Final shapes - Q-values: {q_values.shape}, Targets: {targets.shape}"
        )
        logging.debug(f"Q-values sample: {q_values[:5].tolist()}")
        logging.debug(f"Targets sample: {targets[:5].tolist()}")

        # Ensure shapes match
        assert (
            q_values.shape == targets.shape
        ), f"Shape mismatch: q_values {q_values.shape} vs targets {targets.shape}"

        loss = nn.functional.smooth_l1_loss(q_values, targets)
        logging.debug(f"Computed loss: {loss.item()}")

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        episode_rewards = []
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            for step in range(max_steps):
                state_tensor = torch.FloatTensor(state).to(self.device)
                action = self.select_action(state_tensor, training=True)
                next_state, reward, done, truncated, _ = env.step(action)
                self.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward

                if len(self.replay_buffer) > self.replay_buffer.batch_size:
                    batch = self.replay_buffer.sample(self.replay_buffer.batch_size)
                    loss = self.update(batch)

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            if (episode + 1) % 10 == 0:
                avg_reward = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
                logging.info(
                    f"Episode {episode + 1}, Avg Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.2f}"
                )

        self.is_trained = True
        self.performance = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
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

    def heal(self, env, num_episodes: int, max_steps: int):
        issues = self.diagnose()
        min_episodes = max(
            5, num_episodes // 10
        )  # Ensure at least 5 episodes for healing
        for issue in issues:
            if (
                issue == "Model is not trained"
                or issue == "Model performance is below threshold"
            ):
                logging.info(f"Healing issue: {issue}")
                self.train(env, num_episodes, max_steps)
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(
                    env, min_episodes, max_steps
                )  # Perform a shorter training session

    def update_model(self, env, num_episodes: int, max_steps: int):
        num_episodes = max(1, num_episodes)  # Ensure at least 1 episode
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info["final_reward"]
        self.last_update = time.time()


def create_self_curing_rl_agent(
    features: List[int], action_dim: int
) -> SelfCuringRLAgent:
    return SelfCuringRLAgent(features=features, action_dim=action_dim)


if __name__ == "__main__":
    from .rl_module import RLEnvironment

    logging.basicConfig(level=logging.INFO)
    env = RLEnvironment("CartPole-v1")
    agent = create_self_curing_rl_agent([64, 64], env.action_space.n)

    # Initial training
    training_info = agent.train(env, num_episodes=1000, max_steps=500)
    logging.info(
        f"Initial training completed. Final reward: {training_info['final_reward']}"
    )

    # Simulate some time passing and performance degradation
    agent.last_update -= 100000  # Simulate 27+ hours passing
    agent.performance = 0.7  # Simulate performance drop

    # Diagnose and heal
    issues = agent.diagnose()
    if issues:
        logging.info(f"Detected issues: {issues}")
        agent.heal(env, num_episodes=500, max_steps=500)
        logging.info(f"Healing completed. New performance: {agent.performance}")
