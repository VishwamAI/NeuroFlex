import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
from flax.training import train_state
import optax
import logging
import time
from .rl_module import ReplayBuffer, create_train_state, select_action, train_rl_agent


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
    batch_size: int = 32
    epsilon: float = epsilon_start  # Initialize epsilon as a class attribute

    def setup(self):
        self.q_network = nn.Sequential(
            [nn.Dense(feat) for feat in self.features] + [nn.Dense(self.action_dim)]
        )
        self.target_network = nn.Sequential(
            [nn.Dense(feat) for feat in self.features] + [nn.Dense(self.action_dim)]
        )
        self.replay_buffer = ReplayBuffer(100000)
        self.is_trained = self.variable(
            "agent_state", "is_trained", lambda: jnp.array(False)
        )
        self.performance = self.variable(
            "agent_state", "performance", lambda: jnp.array(0.0)
        )
        self.last_update = self.variable(
            "agent_state", "last_update", lambda: jnp.array(time.time())
        )

    def __call__(self, x):
        return self.q_network(x)

    def select_action(self, params, state: jnp.ndarray, training: bool = False) -> int:
        if training and jax.random.uniform(jax.random.PRNGKey(0)) < self.epsilon:
            return jax.random.randint(jax.random.PRNGKey(0), (), 0, self.action_dim)
        else:
            q_values = self.apply({"params": params}, state)
            return jnp.argmax(q_values)

    def update(
        self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]
    ) -> Tuple[train_state.TrainState, float]:
        def loss_fn(params):
            q_values = self.apply({"params": params}, batch["observations"])
            next_q_values = self.apply(
                {"params": self.target_params}, batch["next_observations"]
            )
            targets = batch["rewards"] + self.gamma * jnp.max(
                next_q_values, axis=-1
            ) * (1 - batch["dones"])
            loss = jnp.mean(
                optax.huber_loss(
                    q_values[jnp.arange(len(batch["actions"])), batch["actions"]],
                    targets,
                )
            )
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        # Update target network
        tau = 0.005  # Soft update parameter
        self.target_params = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau), state.params, self.target_params
        )

        return state, loss

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        rng = jax.random.PRNGKey(0)
        input_shape = (1, env.observation_space.shape[0])
        variables = self.init(rng, jnp.ones(input_shape))
        state = train_state.TrainState.create(
            apply_fn=self.apply,
            params=variables["params"],
            tx=optax.adam(self.learning_rate),
        )
        self.target_params = variables["params"]  # Store target_params separately

        def train_step(carry, _):
            state, env_state, total_reward, step = carry
            action = self.select_action(state.params, env_state, training=True)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            self.replay_buffer.push(env_state, action, reward, next_state, done)

            if len(self.replay_buffer) >= self.batch_size:
                batch = self.replay_buffer.sample(self.batch_size)
                state, _ = self.update(state, batch)

            # Update epsilon
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

            return (state, next_state, total_reward, step + 1), (total_reward, done)

        episode_rewards = []
        for _ in range(num_episodes):
            env_state = env.reset()
            if isinstance(env_state, tuple):
                env_state = env_state[0]
            (state, _, total_reward, _), _ = jax.lax.scan(
                train_step, (state, env_state, 0.0, 0), None, length=max_steps
            )
            episode_rewards.append(total_reward)

        self.is_trained.value = jnp.array(True)
        num_episodes_for_average = max(5, num_episodes // 10)
        recent_rewards = episode_rewards[-num_episodes_for_average:]
        average_reward = jnp.array(sum(recent_rewards) / len(recent_rewards))
        self.performance.value = average_reward
        self.last_update.value = jnp.array(time.time())

        return {
            "is_trained": self.is_trained.value,
            "performance": self.performance.value,
            "episode_rewards": episode_rewards,
            "params": state.params,
            "target_params": self.target_params,
        }

    def diagnose(self) -> List[str]:
        issues = []
        if not self.is_trained.value:
            issues.append("Model is not trained")
        if self.performance.value < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update.value > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")
        return issues

    def heal(self, env, num_episodes: int, max_steps: int):
        issues = self.diagnose()
        for issue in issues:
            if (
                issue == "Model is not trained"
                or issue == "Model performance is below threshold"
            ):
                logging.info(f"Healing issue: {issue}")
                training_info = self.train(env, num_episodes, max_steps)
                self.performance.value = training_info["performance"]
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(
                    env, num_episodes // 10, max_steps
                )  # Perform a shorter training session

    def update_model(self, env, num_episodes: int, max_steps: int):
        training_info = self.train(env, num_episodes, max_steps)
        self.performance.value = training_info["performance"]
        self.last_update.value = jnp.array(time.time())
        return training_info


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
