import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Sequence, Callable
import gym
from flax.training import train_state
import optax
import logging
from collections import deque
import random
import numpy as np
import time
from functools import partial

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.max_priority = 1.0
        self.next_index = 0

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.next_index] = experience
        self.priorities[self.next_index] = self.max_priority
        self.next_index = (self.next_index + 1) % self.capacity

    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)] ** self.alpha
        else:
            probs = self.priorities ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)

        states, actions, rewards, next_states, dones = zip(*samples)
        return {
            'observations': np.array(states),
            'actions': np.array(actions),
            'rewards': np.array(rewards),
            'next_observations': np.array(next_states),
            'dones': np.array(dones, dtype=np.float32),
            'weights': weights,
            'indices': indices
        }

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, np.max(priorities))

    def __len__(self):
        return len(self.buffer)

class RLAgent(nn.Module):
    features: List[int]
    action_dim: int
    is_training: bool = True
    use_double_dqn: bool = True

    @nn.compact
    def __call__(self, x, train: bool = None):
        if train is None:
            train = self.is_training

        for i, feat in enumerate(self.features[:-1]):
            x = nn.Dense(feat, name=f'dense_{i}')(x)
            x = nn.LayerNorm(name=f'layer_norm_{i}')(x)  # Replace BatchNorm with LayerNorm
            x = nn.relu(x)

        # Dueling DQN architecture
        x = nn.relu(nn.Dense(self.features[-1], name='final_dense')(x))
        state_value = nn.Dense(1, name='state_value')(x)
        advantage = nn.Dense(self.action_dim, name='advantage')(x)

        return state_value + (advantage - jnp.mean(advantage, axis=-1, keepdims=True))

    def create_target(self):
        return RLAgent(features=self.features, action_dim=self.action_dim, use_double_dqn=self.use_double_dqn)

    def set_training(self, is_training: bool):
        self.is_training = is_training

    def initialize_layer_stats(self, rng, input_shape):
        dummy_input = jnp.ones((1,) + input_shape)
        variables = self.init(rng, dummy_input, train=True)
        return variables.get('params', {})

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

from flax.training import train_state
from dataclasses import dataclass

@dataclass
class ExtendedTrainState:
    train_state: train_state.TrainState
    batch_stats: dict
    apply_fn: Callable

def create_train_state(rng, model, dummy_input, learning_rate=1e-3):
    if isinstance(dummy_input, tuple):
        dummy_input = dummy_input[0]  # Extract observation from tuple
    dummy_input = jnp.array(dummy_input)  # Ensure input is a JAX array
    variables = model.init(rng, dummy_input[None, ...])
    params, batch_stats = variables['params'], variables.get('batch_stats', {})
    tx = optax.adam(learning_rate)
    base_train_state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)
    return ExtendedTrainState(train_state=base_train_state, batch_stats=batch_stats, apply_fn=model.apply)

def select_action(state: ExtendedTrainState, observation: jnp.ndarray) -> int:
    logging.debug(f"select_action input - state: {state}, observation shape: {observation.shape}")

    action_values, updated_batch_stats = state.train_state.apply_fn(
        {'params': state.train_state.params, 'batch_stats': state.batch_stats},
        observation[None, ...],
        mutable=['batch_stats'],
        train=False
    )
    logging.debug(f"Action values: {action_values}, Updated batch stats: {updated_batch_stats}")

    # Update the batch_stats in the ExtendedTrainState directly
    state.batch_stats = updated_batch_stats['batch_stats']
    logging.debug(f"Updated state batch_stats: {state.batch_stats}")

    selected_action = jnp.argmax(action_values[0])
    logging.debug(f"Selected action: {selected_action}")

    return selected_action

def train_rl_agent(
    agent: RLAgent,
    env: RLEnvironment,
    num_episodes: int = 5000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.995,
    early_stop_threshold: float = 195.0,
    early_stop_episodes: int = 100,
    learning_rate: float = 1e-4,
    target_update_freq: int = 100,
    batch_size: int = 64,
    buffer_size: int = 100000,
    validation_episodes: int = 20,
    min_episodes: int = 1000,
    max_episodes_without_improvement: int = 500,
    validation_threshold: float = 180.0,
    seed: int = 0,
    improvement_threshold: float = 1.005,
    max_training_time: int = 36000,
    n_step_returns: int = 3,
    double_dqn: bool = True
) -> Tuple[train_state.TrainState, List[float], Dict[str, Any]]:
    rng = jax.random.PRNGKey(seed)
    logging.info(f"Initializing RL agent training with seed {seed}")
    start_time = time.time()

    try:
        dummy_obs, _ = env.reset()
        dummy_obs = jnp.array(dummy_obs)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 10,
            peak_value=learning_rate,
            warmup_steps=num_episodes // 10,
            decay_steps=num_episodes,
            end_value=learning_rate / 100
        )
        tx = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=lr_schedule)
        )
        state = create_train_state(rng, agent, dummy_obs, tx)
        target_state = create_train_state(rng, agent, dummy_obs, tx)
        replay_buffer = PrioritizedReplayBuffer(buffer_size)
        logging.info(f"Initialized agent, target network, and prioritized replay buffer with size {buffer_size}")
        logging.info(f"Agent architecture: {agent}")
        logging.info(f"Training parameters: LR={learning_rate}, Gamma={gamma}, Epsilon={epsilon_start}->{epsilon_end}")
        logging.info(f"Episodes={num_episodes}, Max steps={max_steps}, Batch size={batch_size}")
    except Exception as e:
        logging.error(f"Error initializing RL agent: {str(e)}")
        raise RuntimeError(f"Failed to initialize RL agent: {str(e)}")

    @jax.jit
    def update_step(state: train_state.TrainState, target_state: train_state.TrainState, batch: Dict[str, jnp.ndarray], importance_weights: jnp.ndarray, double_dqn: bool = True):
        def loss_fn(params):
            q_values, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['observations'], mutable=['batch_stats'])
            next_q_values_online, _ = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, batch['next_observations'], mutable=['batch_stats'])
            next_q_values_target, _ = target_state.apply_fn({'params': target_state.params, 'batch_stats': target_state.batch_stats}, batch['next_observations'], mutable=False)

            if double_dqn:
                next_actions = jnp.argmax(next_q_values_online, axis=1)
                next_q_values = next_q_values_target[jnp.arange(batch_size), next_actions]
            else:
                next_q_values = jnp.max(next_q_values_target, axis=1)

            targets = batch['rewards'] + gamma * next_q_values * (1 - batch['dones'])
            td_errors = q_values[jnp.arange(batch_size), batch['actions']] - targets
            losses = optax.huber_loss(q_values[jnp.arange(batch_size), batch['actions']], targets)
            loss = jnp.mean(importance_weights * losses)
            return loss, (td_errors, new_model_state)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (td_errors, new_model_state)), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        new_state = new_state.replace(batch_stats=new_model_state['batch_stats'])
        return new_state, loss, td_errors

    episode_rewards, shaped_rewards, recent_losses = [], [], []
    epsilon, best_average_reward = epsilon_start, float('-inf')
    episodes_without_improvement, total_steps = 0, 0
    solved, errors = False, []
    training_info = {'lr_history': [], 'epsilon_history': [], 'episode_lengths': [],
                     'loss_history': [], 'shaped_rewards': [], 'reward_history': [], 'q_values': []}

    logging.info(f"Starting training for {num_episodes} episodes, max {max_steps} steps per episode")
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = jnp.array(obs)
            episode_reward = episode_shaped_reward = 0
            episode_losses, episode_q_values = [], []

            for step in range(max_steps):
                total_steps += 1
                epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))

                rng, action_rng = jax.random.split(rng)
                if jax.random.uniform(action_rng) < epsilon:
                    action = env.action_space.sample()
                    logging.debug(f"Episode {episode + 1}, Step {step + 1}: Random action {action}")
                else:
                    q_values = state.apply_fn({'params': state.params}, obs[None, ...])
                    action = int(jnp.argmax(q_values))
                    episode_q_values.append(float(jnp.max(q_values)))
                    logging.debug(f"Episode {episode + 1}, Step {step + 1}: Greedy action {action}, Q-value: {float(jnp.max(q_values)):.4f}")

                next_obs, reward, done, truncated, _ = env.step(action)
                next_obs = jnp.array(next_obs)
                shaped_reward = reward + 0.01 * (1 - abs(next_obs[2])) + 0.005 * (1 - abs(next_obs[3])) + 0.002 * (1 - abs(next_obs[0]))
                episode_shaped_reward += shaped_reward
                replay_buffer.add(obs, action, shaped_reward, next_obs, done or truncated)

                if len(replay_buffer) >= batch_size:
                    try:
                        batch = replay_buffer.sample(batch_size)
                        state, loss = update_step(state, target_state, jax.tree_map(jnp.array, batch))
                        if jnp.isnan(loss) or jnp.isinf(loss):
                            raise ValueError(f"NaN or Inf loss detected: {loss}")
                        episode_losses.append(float(loss))
                        logging.debug(f"Episode {episode + 1}, Step {step + 1}: Loss: {float(loss):.4f}")
                    except Exception as e:
                        logging.error(f"Error during update step: {str(e)}")
                        errors.append(f"Update step error: {str(e)}")
                        state = state.replace(opt_state=state.tx.init(state.params))
                        epsilon = min(epsilon * 1.01, 1.0)
                        logging.warning(f"Resetting optimizer state and increasing epsilon to {epsilon:.4f}")

                if total_steps % target_update_freq == 0:
                    target_state = target_state.replace(params=state.params)
                    logging.debug(f"Updated target network at step {total_steps}")

                episode_reward += reward
                obs = next_obs
                if done or truncated:
                    logging.debug(f"Episode {episode + 1} ended after {step + 1} steps")
                    break

            episode_rewards.append(episode_reward)
            shaped_rewards.append(episode_shaped_reward)
            current_lr = lr_schedule(episode)

            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
            avg_shaped_reward = sum(shaped_rewards[-100:]) / min(len(shaped_rewards), 100)
            avg_loss = sum(episode_losses) / max(len(episode_losses), 1)
            avg_q_value = sum(episode_q_values) / max(len(episode_q_values), 1)
            recent_losses.append(avg_loss)

            logging.info(f"Episode {episode + 1}/{num_episodes}, Steps: {step + 1}, "
                         f"Reward: {episode_reward:.2f}, Shaped: {episode_shaped_reward:.2f}, "
                         f"Avg Reward: {avg_reward:.2f}, Avg Shaped: {avg_shaped_reward:.2f}, "
                         f"Avg Loss: {avg_loss:.4f}, Epsilon: {epsilon:.4f}, LR: {current_lr:.6f}, "
                         f"Avg Q-value: {avg_q_value:.4f}")

            for key, value in zip(
                ['lr_history', 'epsilon_history', 'episode_lengths', 'loss_history', 'shaped_rewards', 'reward_history', 'q_values'],
                [current_lr, epsilon, step + 1, avg_loss, episode_shaped_reward, episode_reward, avg_q_value]
            ):
                training_info[key].append(value)

            if avg_shaped_reward > best_average_reward * improvement_threshold:
                best_average_reward = avg_shaped_reward
                episodes_without_improvement = 0
                logging.info(f"New best average shaped reward: {best_average_reward:.2f}")
            else:
                episodes_without_improvement += 1
                logging.debug(f"Episodes without improvement: {episodes_without_improvement}")

            if len(recent_losses) >= 50:
                loss_ratio = max(recent_losses[-50:]) / (min(recent_losses[-50:]) + 1e-8)
                if loss_ratio > 3 or jnp.isnan(jnp.array(recent_losses[-50:])).any():
                    logging.warning(f"Detected training instability. Loss ratio: {loss_ratio:.2f}")
                    state = state.replace(opt_state=state.tx.init(state.params))
                    epsilon = min(epsilon * 1.02, 1.0)
                    episodes_without_improvement = 0
                    logging.info(f"Reset optimizer state and increased epsilon to {epsilon:.4f}")

            if avg_shaped_reward >= early_stop_threshold and episode >= min_episodes:
                validation_rewards = run_validation(state, env, validation_episodes, max_steps)
                val_avg_reward = sum(validation_rewards) / validation_episodes
                val_std_reward = np.std(validation_rewards)
                if val_avg_reward >= validation_threshold and val_std_reward < early_stop_threshold * 0.25:
                    logging.info(f"Environment solved! Validation avg: {val_avg_reward:.2f}, std: {val_std_reward:.2f}")
                    solved = True
                    training_info['early_stop_reason'] = 'solved'
                    break
                else:
                    logging.info(f"Validation failed. Avg: {val_avg_reward:.2f}, std: {val_std_reward:.2f}")
                    episodes_without_improvement = 0

            if episodes_without_improvement >= early_stop_episodes:
                logging.info(f"Early stopping: no improvement for {early_stop_episodes} episodes")
                training_info['early_stop_reason'] = 'no_improvement'
                break

            if episodes_without_improvement >= max_episodes_without_improvement:
                logging.warning(f"No improvement for {max_episodes_without_improvement} episodes. Stopping.")
                training_info['early_stop_reason'] = 'max_episodes_without_improvement'
                break

            if episode % (num_episodes // 20) == 0:
                validation_rewards = run_validation(state, env, validation_episodes, max_steps)
                val_avg_reward = sum(validation_rewards) / validation_episodes
                logging.info(f"Periodic validation - Average reward: {val_avg_reward:.2f}")
                if val_avg_reward < best_average_reward * 0.9:
                    logging.warning("Significant performance drop. Adjusting epsilon and learning rate.")
                    epsilon = min(epsilon * 1.05, 1.0)
                    current_lr *= 0.9
                    state = state.replace(tx=optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adam(learning_rate=current_lr)
                    ))
                    logging.info(f"Adjusted epsilon to {epsilon:.4f} and learning rate to {current_lr:.6f}")

            if time.time() - start_time > max_training_time:
                logging.warning(f"Maximum training time of {max_training_time} seconds reached. Stopping.")
                training_info['early_stop_reason'] = 'max_training_time_reached'
                break

    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}")
        errors.append(f"Training error: {str(e)}")
        training_info['training_stopped_early'] = True

    if not episode_rewards:
        raise ValueError("No episodes completed successfully")

    logging_message = "Training completed successfully" if solved else "Training completed without solving"
    logging.info(f"{logging_message}. Best average reward: {best_average_reward:.2f}")

    training_info.update({
        'final_lr': current_lr,
        'best_average_reward': best_average_reward,
        'total_episodes': episode + 1,
        'total_steps': total_steps,
        'solved': solved,
        'errors': errors,
        'validation_rewards': validation_rewards if 'validation_rewards' in locals() else None,
        'training_stopped_early': episodes_without_improvement >= max_episodes_without_improvement,
        'total_training_time': time.time() - start_time
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
    trained_state, rewards, info = train_rl_agent(agent, env, num_episodes=100, max_steps=500)
    print(f"Average reward over last 10 episodes: {sum(rewards[-10:]) / 10:.2f}")
    print(f"Training info: {info}")
