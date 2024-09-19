import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
from flax.training import train_state
import optax
import logging
import time
from .rl_module import ReplayBuffer, create_train_state, select_action, train_rl_agent
from sklearn.ensemble import IsolationForest
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
import numpy as np
from scipy.stats import entropy
from dataclasses import field

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
    performance_history: List[float] = field(default_factory=list)  # Initialize performance_history as an empty list

    def setup(self):
        self.q_network = nn.Sequential([
            nn.Dense(self.features[0]),
            *[nn.Dense(feat) for feat in self.features[1:]],
            nn.Dense(self.action_dim)
        ])
        self.target_network = self.q_network.clone()  # Sharing the initial weights
        self.replay_buffer = ReplayBuffer(100000)
        self.input_shape = (4,)  # Assuming CartPole-v1 environment with 4 state dimensions

    def __call__(self, x, train=False, mutable=['agent_state']):
        if not hasattr(self, 'agent_state'):
            self.agent_state = {
                'is_trained': jnp.array(False),
                'performance': jnp.array(0.0),
                'last_update': jnp.array(time.time()),
                'anomaly_detector': IsolationForest(contamination=0.1),
                'causal_model': BayesianNetwork([('state', 'action'), ('action', 'reward')]),
                'action_history': []
            }
        if not hasattr(self, 'replay_buffer'):
            self.replay_buffer = ReplayBuffer(100000)
        return self.q_network(x)

    def __call__(self, x):
        return self.q_network(x)

    def select_action(self, params, state: jnp.ndarray, key: jnp.ndarray, training: bool = False) -> int:
        action = jax.lax.cond(
            training & (jax.random.uniform(key) < self.epsilon),
            lambda _: jax.random.randint(key, (), 0, self.action_dim),
            lambda _: jnp.argmax(self.apply({'params': params}, state)),
            operand=None
        )
        return jax.numpy.asarray(action, dtype=int)

    def update(self, state: train_state.TrainState, batch: Dict[str, jnp.ndarray]) -> Tuple[train_state.TrainState, float]:
        def loss_fn(params):
            q_values = self.apply({'params': params}, batch['observations'])
            next_q_values = self.apply({'params': self.target_params}, batch['next_observations'])
            targets = batch['rewards'] + self.gamma * jnp.max(next_q_values, axis=-1) * (1 - batch['dones'])
            loss = jnp.mean(optax.huber_loss(q_values[jnp.arange(len(batch['actions'])), batch['actions']], targets))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)

        # Update target network
        tau = 0.005  # Soft update parameter
        self.target_params = jax.tree_map(
            lambda p, tp: p * tau + tp * (1 - tau),
            state.params, self.target_params
        )

        return state, loss

    def train(self, env, num_episodes: int, max_steps: int) -> Dict[str, Any]:
        rng = jax.random.PRNGKey(0)
        input_shape = env.observation_space.shape
        variables = self.init(rng, jnp.ones(input_shape))
        state = train_state.TrainState.create(
            apply_fn=self.apply,
            params=variables['params'],
            tx=optax.adam(self.learning_rate)
        )
        self.target_params = variables['params']  # Store target_params separately
        replay_buffer = ReplayBuffer(100000)  # Initialize replay buffer here

        def train_step(carry, _):
            state, env_state, total_reward, step, key, epsilon = carry
            key, subkey = jax.random.split(key)

            def select_action(params, state, key):
                return self.select_action(params, state, key, training=True)

            action = jax.lax.cond(
                jax.random.uniform(subkey) < epsilon,
                lambda _: jax.random.randint(subkey, (), 0, self.action_dim),
                lambda _: select_action(state.params, env_state, subkey),
                operand=None
            )

            # Use jax.numpy operations to ensure action is a concrete integer
            action_int = jnp.asarray(action, dtype=jnp.int32)

            # Use jax.pure_callback to interact with the environment
            def env_step(action):
                next_state, reward, done, _ = env.step(int(action))
                return jnp.array(next_state, dtype=jnp.float32), jnp.array(reward, dtype=jnp.float32), jnp.array(done, dtype=jnp.bool_)

            next_state, reward, done = jax.pure_callback(env_step, (jnp.zeros_like(env_state), jnp.array(0., dtype=jnp.float32), jnp.array(False)), action_int)

            total_reward += reward

            # Update epsilon
            new_epsilon = jnp.maximum(
                self.epsilon_end,
                epsilon * self.epsilon_decay
            )

            transition = (env_state, action_int, reward, next_state, done)
            return (state, next_state, total_reward, step + 1, key, new_epsilon), transition

        episode_rewards = []
        for _ in range(num_episodes):
            env_state = env.reset()
            if isinstance(env_state, tuple):
                env_state = env_state[0]
            env_state = jnp.array(env_state, dtype=jnp.float32)  # Ensure env_state is a JAX array
            rng, subkey = jax.random.split(rng)
            (state, _, total_reward, _, _, _), transitions = jax.lax.scan(
                train_step, (state, env_state, jnp.array(0.0), jnp.array(0), subkey, self.epsilon), None, length=max_steps
            )
            episode_rewards.append(total_reward)

            # Update replay buffer outside of JAX loop
            transitions = jax.tree_util.tree_map(lambda x: x.reshape(-1, *x.shape[2:]), transitions)
            for transition in zip(*transitions):
                env_state, action, reward, next_state, done = transition
                replay_buffer.push(env_state, action, reward, next_state, done)

            # Perform batch updates
            if len(replay_buffer) >= self.batch_size:
                batch = replay_buffer.sample(self.batch_size)
                state, _ = self.update(state, batch)

        # Set is_trained based on number of episodes and performance threshold
        num_episodes_for_average = max(5, num_episodes // 10)
        recent_rewards = episode_rewards[-num_episodes_for_average:]
        average_reward = jnp.mean(jnp.array(recent_rewards))
        self.performance = average_reward
        self.is_trained = jnp.array(num_episodes >= 10 and self.performance >= 100.0)
        self.last_update = jnp.array(time.time())
        self.performance_history.append(average_reward)

        return {
            'is_trained': self.is_trained,
            'performance': self.performance,
            'episode_rewards': episode_rewards,
            'params': state.params,
            'target_params': self.target_params
        }

    def diagnose(self) -> List[str]:
        issues = []
        if not self.is_trained.value:
            issues.append("Model is not trained")
        if self.performance.value < self.performance_threshold:
            issues.append("Model performance is below threshold")
        if time.time() - self.last_update.value > self.update_interval:
            issues.append("Model hasn't been updated in 24 hours")

        # Anomaly detection
        if len(self.performance_history) > 10:
            recent_performance = np.array(self.performance_history[-10:]).reshape(-1, 1)
            anomaly_scores = self.anomaly_detector.fit_predict(recent_performance)
            if -1 in anomaly_scores:
                issues.append("Anomaly detected in recent performance")

        # Action distribution analysis
        if len(self.action_history) > 100:
            action_counts = np.bincount(self.action_history[-100:], minlength=self.action_dim)
            action_probs = action_counts / len(self.action_history[-100:])
            action_entropy = entropy(action_probs)
            if action_entropy < 0.5:  # Threshold for low entropy
                issues.append("Low action diversity detected")

        return issues

    def heal(self, env, num_episodes: int, max_steps: int):
        issues = self.diagnose()
        for issue in issues:
            if issue == "Model is not trained" or issue == "Model performance is below threshold":
                logging.info(f"Healing issue: {issue}")
                training_info = self.train(env, num_episodes, max_steps)
                self.performance.value = training_info['performance']
            elif issue == "Model hasn't been updated in 24 hours":
                logging.info(f"Healing issue: {issue}")
                self.update_model(env, num_episodes // 10, max_steps)  # Perform a shorter training session
            elif issue == "Anomaly detected in recent performance":
                logging.info(f"Healing issue: {issue}")
                self.reset_and_retrain(env, num_episodes, max_steps)
            elif issue == "Low action diversity detected":
                logging.info(f"Healing issue: {issue}")
                self.increase_exploration(env, num_episodes // 5, max_steps)

    def update_model(self, env, num_episodes: int, max_steps: int):
        training_info = self.train(env, num_episodes, max_steps)
        self.performance.value = training_info['performance']
        self.last_update.value = jnp.array(time.time())
        return training_info

    def reset_and_retrain(self, env, num_episodes: int, max_steps: int):
        self.q_network = nn.Sequential([
            nn.Dense(feat) for feat in self.features
        ] + [nn.Dense(self.action_dim)])
        self.target_network = nn.Sequential([
            nn.Dense(feat) for feat in self.features
        ] + [nn.Dense(self.action_dim)])
        self.epsilon = self.epsilon_start
        return self.train(env, num_episodes, max_steps)

    def increase_exploration(self, env, num_episodes: int, max_steps: int):
        original_epsilon = self.epsilon
        self.epsilon = min(self.epsilon * 2, 1.0)
        training_info = self.train(env, num_episodes, max_steps)
        self.epsilon = original_epsilon
        return training_info

def create_self_curing_rl_agent(features: List[int], action_dim: int) -> SelfCuringRLAgent:
    return SelfCuringRLAgent(features=features, action_dim=action_dim)

if __name__ == "__main__":
    from .rl_module import RLEnvironment

    logging.basicConfig(level=logging.INFO)
    env = RLEnvironment("CartPole-v1")
    agent = create_self_curing_rl_agent([64, 64], env.action_space.n)

    # Initial training
    training_info = agent.train(env, num_episodes=1000, max_steps=500)
    logging.info(f"Initial training completed. Final reward: {training_info['performance']}")

    # Simulate some time passing and performance degradation
    agent.last_update.value -= 100000  # Simulate 27+ hours passing
    agent.performance.value = jnp.array(0.7)  # Simulate performance drop

    # Diagnose and heal
    issues = agent.diagnose()
    if issues:
        logging.info(f"Detected issues: {issues}")
        agent.heal(env, num_episodes=500, max_steps=500)
        logging.info(f"Healing completed. New performance: {agent.performance}")
