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
    num_episodes: int = 2000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    early_stop_threshold: float = 195.0,
    early_stop_episodes: int = 100,
    learning_rate: float = 1e-3,
    target_update_freq: int = 100,
    batch_size: int = 64,
    buffer_size: int = 10000,
    validation_episodes: int = 10,
    min_episodes: int = 500,
    max_episodes_without_improvement: int = 200,
    seed: int = 0
) -> Tuple[train_state.TrainState, List[float], Dict[str, Any]]:
    rng = jax.random.PRNGKey(seed)
    try:
        dummy_obs, _ = env.reset()
        state = create_train_state(rng, agent, dummy_obs[None, ...], learning_rate)
        target_state = create_train_state(rng, agent, dummy_obs[None, ...], learning_rate)
        replay_buffer = ReplayBuffer(buffer_size)
    except Exception as e:
        logging.error(f"Error initializing RL agent: {str(e)}")
        raise

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
    training_info = {'lr_history': [], 'epsilon_history': [], 'episode_lengths': [], 'loss_history': [], 'shaped_rewards': []}

    # Improved learning rate scheduler with warm-up and cosine decay
    warmup_steps = num_episodes // 20
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=learning_rate * 0.1,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=num_episodes - warmup_steps,
        end_value=learning_rate * 0.01
    )

    loss_history = []

    for episode in range(num_episodes):
        try:
            obs, _ = env.reset()
            episode_reward = 0
            episode_losses = []
            episode_steps = 0
            episode_shaped_reward = 0

            for step in range(max_steps):
                total_steps += 1
                episode_steps += 1

                # Improved exploration strategy with epsilon decay
                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
                if jax.random.uniform(jax.random.PRNGKey(total_steps)) < epsilon:
                    action = env.action_space.sample()
                else:
                    q_values = state.apply_fn({'params': state.params}, obs[None, ...])
                    action = int(jnp.argmax(q_values))

                next_obs, reward, done, truncated, _ = env.step(action)

                # Implement reward shaping
                shaped_reward = reward + 0.1 * (1 - abs(next_obs[2])) + 0.05 * (1 - abs(next_obs[3]))  # Encourage upright position and low angular velocity
                episode_shaped_reward += shaped_reward

                replay_buffer.add(obs, action, shaped_reward, next_obs, done or truncated)

                if len(replay_buffer) >= batch_size:
                    batch = replay_buffer.sample(batch_size)
                    state, loss = update_step(state, target_state, batch)
                    episode_losses.append(loss)

                # Log shaped reward for debugging
                logging.debug(f"Step {episode_steps}: Reward={reward:.2f}, Shaped Reward={shaped_reward:.2f}")

                if total_steps % target_update_freq == 0:
                    target_state = target_state.replace(params=state.params)

                episode_reward += reward
                obs = next_obs

                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            shaped_rewards.append(episode_shaped_reward)

            # Update learning rate with warm-up and cosine decay
            current_lr = lr_schedule(episode)
            state = state.replace(tx=optax.adam(current_lr))

            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            avg_shaped_reward = sum(shaped_rewards[-100:]) / min(len(shaped_rewards), 100)
            avg_loss = sum(episode_losses) / len(episode_losses) if episode_losses else 0
            recent_losses.append(avg_loss)
            loss_history.append(avg_loss)
            if len(recent_losses) > 100:
                recent_losses.pop(0)

            # Enhanced logging
            logging.info(f"Episode {episode + 1}/{num_episodes}, Steps: {episode_steps}, "
                         f"Reward: {episode_reward:.2f}, Shaped Reward: {episode_shaped_reward:.2f}, "
                         f"Avg Reward: {avg_reward:.2f}, Avg Shaped Reward: {avg_shaped_reward:.2f}, "
                         f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}, LR: {current_lr:.6f}")
            logging.debug(f"Loss history: {loss_history[-10:]}")  # Log last 10 loss values
            logging.debug(f"Shaped rewards: {shaped_rewards[-10:]}")  # Log last 10 shaped rewards

            # Update training info
            training_info['lr_history'].append(current_lr)
            training_info['epsilon_history'].append(epsilon)
            training_info['episode_lengths'].append(episode_steps)
            training_info['shaped_rewards'].append(episode_shaped_reward)
            training_info['loss_history'].append(avg_loss)
            training_info['reward_history'].append(episode_reward)

            if avg_shaped_reward > best_average_reward:
                best_average_reward = avg_shaped_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1

            # Improved training instability detection
            if len(recent_losses) >= 10:
                loss_ratio = max(recent_losses[-10:]) / (min(recent_losses[-10:]) + 1e-8)
                if loss_ratio > 10 or jnp.isnan(jnp.array(recent_losses[-10:])).any():
                    logging.warning(f"Detected training instability. Loss ratio: {loss_ratio:.2f}")
                    state = state.replace(opt_state=state.tx.init(state.params))
                    epsilon = min(epsilon * 1.5, 1.0)  # Increase exploration
                    logging.info(f"Resetting optimizer state and increasing epsilon to {epsilon:.4f}")

            # Improved early stopping condition
            if avg_shaped_reward >= early_stop_threshold and episode >= min_episodes:
                logging.info(f"Potential solution found! Running validation...")
                validation_rewards = []
                for _ in range(validation_episodes):
                    val_obs, _ = env.reset()
                    val_reward = 0
                    for _ in range(max_steps):
                        q_values = state.apply_fn({'params': state.params}, val_obs[None, ...])
                        val_action = int(jnp.argmax(q_values))
                        val_obs, r, d, t, _ = env.step(val_action)
                        val_reward += r
                        if d or t:
                            break
                    validation_rewards.append(val_reward)
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
            logging.error(f"Error in episode {episode + 1}: {str(e)}")
            errors.append(str(e))
            training_info['training_stopped_early'] = True
            continue

    if not episode_rewards:
        raise ValueError("No episodes completed successfully")

    if solved:
        logging.info(f"Training completed successfully. Best average reward: {best_average_reward:.2f}")
    else:
        logging.info(f"Training completed without solving the environment. Best average reward: {best_average_reward:.2f}")

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

# Example usage
if __name__ == "__main__":
    env = RLEnvironment("CartPole-v1")
    agent = RLAgent(features=[64, 64], action_dim=env.action_space.n)
    trained_state, rewards = train_rl_agent(agent, env, num_episodes=100, max_steps=500)
    print(f"Average reward over last 10 episodes: {sum(rewards[-10:]) / 10:.2f}")
