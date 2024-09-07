import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
from flax.training import train_state
import optax
import logging
import time
from .rl_module import ReplayBuffer, create_train_state, select_action, train_rl_agent
from ..utils import utils

class SelfCuringRLAgent(nn.Module):
    features: List[int]
    action_dim: int
    learning_rate: float = 1e-4
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    performance_threshold: float = 0.8
    update_interval: int = 86400  # 24 hours in seconds

    def setup(self):
        self.q_network = nn.Sequential([
            nn.Dense(feat) for feat in self.features
        ] + [nn.Dense(self.action_dim)])
        self.optimizer = optax.adam(self.learning_rate)
        self.replay_buffer = ReplayBuffer(100000)
        self.epsilon = self.epsilon_start
        self.is_trained = False
        self.performance = 0.0
        self.last_update = time.time()

    def __call__(self, x):
        return self.q_network(x)

    def select_action(self, state: jnp.ndarray, training: bool = False) -> int:
        if training and jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim)
        else:
            q_values = self(state)
            return jnp.argmax(q_values)

    def update(self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, float]:
        def loss_fn(params):
            q_values = self.apply({'params': params}, batch['observations'])
            next_q_values = self.apply({'params': params}, batch['next_observations'])
            targets = batch['rewards'] + self.gamma * jnp.max(next_q_values, axis=-1) * (1 - batch['dones'])
            loss = jnp.mean(optax.huber_loss(q_values[jnp.arange(len(batch['actions'])), batch['actions']], targets))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        state = create_train_state(jax.random.PRNGKey(0), self, env.observation_space.shape)
        training_info = train_rl_agent(self, env, state, num_episodes, max_steps)
        self.is_trained = True
        self.performance = training_info['final_reward']
        self.last_update = time.time()
        return training_info

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
        for issue in issues:
            if issue == "Model is not trained" or issue == "Model performance is below threshold":
                logging.info(f"Healing issue: {issue}")
                self.train(env, num_episodes, max_steps)
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(env, num_episodes // 10, max_steps)  # Perform a shorter training session

    def update_model(self, env, num_episodes: int, max_steps: int):
        training_info = self.train(env, num_episodes, max_steps)
        self.performance = training_info['final_reward']
        self.last_update = time.time()

def create_self_curing_rl_agent(features: List[int], action_dim: int) -> SelfCuringRLAgent:
    return SelfCuringRLAgent(features=features, action_dim=action_dim)

if __name__ == "__main__":
    from .rl_module import RLEnvironment

    logging.basicConfig(level=logging.INFO)
    env = RLEnvironment("CartPole-v1")
    agent = create_self_curing_rl_agent([64, 64], env.action_space.n)

    # Initial training
    training_info = agent.train(env, num_episodes=1000, max_steps=500)
    logging.info(f"Initial training completed. Final reward: {training_info['final_reward']}")

    # Simulate some time passing and performance degradation
    agent.last_update -= 100000  # Simulate 27+ hours passing
    agent.performance = 0.7  # Simulate performance drop

    # Diagnose and heal
    issues = agent.diagnose()
    if issues:
        logging.info(f"Detected issues: {issues}")
        agent.heal(env, num_episodes=500, max_steps=500)
        logging.info(f"Healing completed. New performance: {agent.performance}")
