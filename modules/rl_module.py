import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, NamedTuple
import gym
from flax.training import train_state
import optax
import logging
from collections import deque
import random
import numpy as np
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return {
            'observations': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_states),
            'dones': np.array(dones, dtype=np.float32)
        }

    def __len__(self):
        return len(self.buffer)

class RLAgent(nn.Module):
    features: List[int]
    action_dim: int

    @nn.compact
    def __call__(self, x):
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.relu(x)
        return nn.Dense(self.action_dim)(x)

    def create_target(self):
        return RLAgent(features=self.features, action_dim=self.action_dim)

class RLEnvironment:
    def __init__(self, env_name: str):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self) -> Tuple[jnp.ndarray, Dict]:
        observation, info = self.env.reset()
        return jnp.array(observation), info

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return jnp.array(obs), reward, terminated, truncated, info

def create_train_state(rng, model, dummy_input, learning_rate=1e-3):
    if isinstance(dummy_input, tuple):
        dummy_input = dummy_input[0]  # Extract observation from tuple
    dummy_input = jnp.array(dummy_input)  # Ensure input is a JAX array
    params = model.init(rng, dummy_input[None, ...])['params']
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def select_action(state: train_state.TrainState, observation: jnp.ndarray) -> int:
    action_values = state.apply_fn({'params': state.params}, observation)
    return jnp.argmax(action_values)

def train_rl_agent(
    agent: RLAgent,
    env: RLEnvironment,
    num_episodes: int = 5000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.997,
    early_stop_threshold: float = 195.0,
    early_stop_episodes: int = 100,
    learning_rate: float = 5e-4,
    target_update_freq: int = 100,
    batch_size: int = 128,
    buffer_size: int = 100000,
    validation_episodes: int = 10,
    min_episodes: int = 1000,
    max_episodes_without_improvement: int = 300,
    seed: int = 0
) -> Tuple[train_state.TrainState, List[float], Dict[str, Any]]:
    rng = jax.random.PRNGKey(seed)
    logging.info(f"Initializing RL agent training with seed {seed}")

    try:
        dummy_obs, _ = env.reset()
        state = create_train_state(rng, agent, dummy_obs[None, ...], learning_rate)
        target_state = create_train_state(rng, agent, dummy_obs[None, ...], learning_rate)
        replay_buffer = ReplayBuffer(buffer_size)
        logging.info(f"Initialized agent, target network, and replay buffer with size {buffer_size}")
    except Exception as e:
        logging.error(f"Error initializing RL agent: {str(e)}")
        raise RuntimeError(f"Failed to initialize RL agent: {str(e)}")

    @jax.jit
    def update_step(state: train_state.TrainState, target_state: train_state.TrainState, batch: Dict[str, jnp.ndarray]):
        def loss_fn(params):
            q_values = state.apply_fn({'params': params}, batch['observations'])
            next_q_values = target_state.apply_fn({'params': target_state.params}, batch['next_observations'])
            next_actions = jnp.argmax(state.apply_fn({'params': params}, batch['next_observations']), axis=1)
            targets = batch['rewards'] + gamma * next_q_values[jnp.arange(batch_size), next_actions] * (1 - batch['dones'])
            loss = jnp.mean(optax.huber_loss(q_values[jnp.arange(batch_size), batch['actions']], targets))
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        return state.apply_gradients(grads=grads), loss

    episode_rewards = []
    shaped_rewards = []
    epsilon = epsilon_start
    best_average_reward = float('-inf')
    episodes_without_improvement = 0
    solved = False
    total_steps = 0
    recent_losses = []
    errors = []
    training_info = {'lr_history': [], 'epsilon_history': [], 'episode_lengths': [], 'loss_history': [], 'shaped_rewards': [], 'reward_history': [], 'q_values': []}

    warmup_steps = num_episodes // 20
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.1,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_episodes - warmup_steps,
        end_value=learning_rate * 0.01
    )

    logging.info(f"Starting training for {num_episodes} episodes, max {max_steps} steps per episode")
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_losses = []
            episode_steps = 0
            episode_shaped_reward = 0
            episode_q_values = []

            for step in range(max_steps):
                total_steps += 1
                episode_steps += 1

                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
                if jax.random.uniform(jax.random.PRNGKey(total_steps)) < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = state.apply_fn({'params': state.params}, obs[None, ...])
                    action = int(jnp.argmax(q_values))
                    episode_q_values.append(float(jnp.max(q_values)))

                next_obs, reward, done, truncated, _ = env.step(action)

                shaped_reward = reward + 0.2 * (1 - abs(next_obs[2])) + 0.1 * (1 - abs(next_obs[3])) + 0.05 * (1 - abs(next_obs[0]))
                episode_shaped_reward += shaped_reward

                replay_buffer.add(obs, action, shaped_reward, next_obs, done or truncated)

                if len(replay_buffer) >= batch_size:
                    try:
                        batch = replay_buffer.sample(batch_size)
                        state, loss = update_step(state, target_state, batch)
                        if jnp.isnan(loss) or jnp.isinf(loss):
                            raise ValueError(f"NaN or Inf loss detected: {loss}")
                        episode_losses.append(float(loss))
                    except Exception as e:
                        logging.error(f"Error during update step: {str(e)}")
                        errors.append(f"Update step error: {str(e)}")
                        state = state.replace(opt_state=state.tx.init(state.params))
                        logging.info("Reset optimizer state due to error")

                if total_steps % target_update_freq == 0:
                    target_state = target_state.replace(params=state.params)
                    logging.debug(f"Updated target network at step {total_steps}")

                episode_reward += reward
                obs = next_obs

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            shaped_rewards.append(episode_shaped_reward)

            current_lr = lr_schedule(episode)
            state = state.replace(tx=optax.adam(current_lr))

            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            avg_shaped_reward = sum(shaped_rewards[-100:]) / min(len(shaped_rewards), 100)
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            recent_losses.append(avg_loss)

            logging.info(f"Episode {episode + 1}/{num_episodes}, Steps: {episode_steps}, "
                         f"Reward: {episode_reward:.2f}, Shaped Reward: {episode_shaped_reward:.2f}, "
                         f"Avg Reward: {avg_reward:.2f}, Avg Shaped Reward: {avg_shaped_reward:.2f}, "
                         f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}, LR: {current_lr:.6f}, "
                         f"Avg Q-value: {sum(episode_q_values) / len(episode_q_values):.4f}")

            for key, value in zip(
                ['lr_history', 'epsilon_history', 'episode_lengths', 'loss_history', 'shaped_rewards', 'reward_history', 'q_values'],
                [current_lr, epsilon, episode_steps, avg_loss, episode_shaped_reward, episode_reward, sum(episode_q_values) / len(episode_q_values)]
            ):
                training_info[key].append(value)

            if avg_shaped_reward > best_average_reward:
                best_average_reward = avg_shaped_reward
                episodes_without_improvement = 0
                logging.info(f"New best average shaped reward: {best_average_reward:.2f}")
            else:
                episodes_without_improvement += 1

            if len(recent_losses) >= 10:
                loss_ratio = max(recent_losses[-10:]) / (min(recent_losses[-10:]) + 1e-8)
                if loss_ratio > 5 or jnp.isnan(jnp.array(recent_losses[-10:])).any():
                    logging.warning(f"Detected training instability. Loss ratio: {loss_ratio:.2f}")
                    state = state.replace(opt_state=state.tx.init(state.params))
                    epsilon = min(epsilon * 1.2, 1.0)
                    current_lr *= 0.5
                    state = state.replace(tx=optax.adam(current_lr))
                    logging.info(f"Reset optimizer state, increased epsilon to {epsilon:.4f}, and reduced learning rate to {current_lr:.6f}")

            if avg_shaped_reward >= early_stop_threshold and episode >= min_episodes:
                logging.info(f"Potential solution found! Running validation...")
                validation_rewards = run_validation(state, env, validation_episodes, max_steps)
                val_avg_reward = sum(validation_rewards) / validation_episodes
                val_std_reward = np.std(validation_rewards)
                if val_avg_reward >= early_stop_threshold and val_std_reward < early_stop_threshold * 0.1:
                    logging.info(f"Environment solved! Validation average reward: {val_avg_reward:.2f}, std: {val_std_reward:.2f}")
                    solved = True
                    training_info['early_stop_reason'] = 'solved'
                    break
                else:
                    logging.info(f"Validation failed. Avg: {val_avg_reward:.2f}, std: {val_std_reward:.2f}. Continuing training...")

            if episodes_without_improvement >= early_stop_episodes:
                logging.info(f"Early stopping: no improvement for {early_stop_episodes} episodes")
                training_info['early_stop_reason'] = 'no_improvement'
                break

            if episodes_without_improvement >= max_episodes_without_improvement:
                logging.warning(f"No improvement for {max_episodes_without_improvement} episodes. Stopping training.")
                training_info['early_stop_reason'] = 'max_episodes_without_improvement'
                training_info['training_stopped_early'] = True
                break

    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}")
        errors.append(f"Training error: {str(e)}")
        training_info['training_stopped_early'] = True

    if not episode_rewards:
        raise ValueError("No episodes completed successfully")

    logging_message = "Training completed successfully" if solved else "Training completed without solving the environment"
    logging.info(f"{logging_message}. Best average reward: {best_average_reward:.2f}")

    training_info.update({
        'final_lr': lr_schedule(episode),
        'best_average_reward': best_average_reward,
        'total_episodes': episode + 1,
        'total_steps': total_steps,
        'solved': solved,
        'errors': errors,
        'validation_rewards': validation_rewards if 'validation_rewards' in locals() else None,
        'training_stopped_early': episodes_without_improvement >= max_episodes_without_improvement,
    })

    return state, episode_rewards, training_info

def run_validation(state: train_state.TrainState, env: RLEnvironment, num_episodes: int, max_steps: int) -> List[float]:
    validation_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            q_values = state.apply_fn({'params': state.params}, obs[None, ...])
            action = int(jnp.argmax(q_values))
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            if done or truncated:
                break
        validation_rewards.append(episode_reward)
    return validation_rewards

# Example usage
if __name__ == "__main__":
    env = RLEnvironment("CartPole-v1")
    agent = RLAgent(features=[64, 64], action_dim=env.action_space.n)
    trained_state, rewards = train_rl_agent(agent, env, num_episodes=100, max_steps=500)
    print(f"Average reward over last 10 episodes: {sum(rewards[-10:]) / 10:.2f}")
