import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any
import gym
from flax.training import train_state
import optax
import logging
from collections import deque
import random
import numpy as np
import time
import heapq
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4, beta_increment: float = 0.001):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        if len(self.buffer) < self.capacity:
            probs = self.priorities[:len(self.buffer)]
        else:
            probs = self.priorities
        probs = probs ** self.alpha
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
            'dones': np.array(dones, dtype=np.float32)
        }, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

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

class HierarchicalRLAgent(nn.Module):
    features: List[int]
    action_dims: List[int]
    num_levels: int

    @nn.compact
    def __call__(self, x):
        outputs = []
        for level in range(self.num_levels):
            sub_agent = RLAgent(features=self.features, action_dim=self.action_dims[level])
            outputs.append(sub_agent(x))
            x = jnp.concatenate([x, outputs[-1]], axis=-1)
        return outputs

    def create_target(self):
        return HierarchicalRLAgent(features=self.features, action_dims=self.action_dims, num_levels=self.num_levels)

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

    if isinstance(model, HierarchicalRLAgent):
        # Initialize parameters for each sub-agent in the hierarchy
        params = model.init(rng, dummy_input[None, ...])['params']
        tx = optax.adam(learning_rate)
        return [train_state.TrainState.create(
            apply_fn=model.apply, params=p, tx=tx) for p in params]
    else:
        # Regular RLAgent initialization
        params = model.init(rng, dummy_input[None, ...])['params']
        tx = optax.adam(learning_rate)
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def select_action(state: train_state.TrainState, observation: jnp.ndarray) -> int:
    action_values = state.apply_fn({'params': state.params}, observation)
    return jnp.argmax(action_values)

def train_rl_agent(
    agent: HierarchicalRLAgent,
    env: RLEnvironment,
    num_episodes: int = 200000,  # Increased number of episodes
    max_steps: int = 1000,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.01,
    epsilon_decay: float = 0.999995,  # Even slower decay
    early_stop_threshold: float = 195.0,
    early_stop_episodes: int = 5000,  # Increased patience
    learning_rate: float = 5e-5,  # Reduced learning rate for stability
    target_update_freq: int = 2000,  # Increased update frequency
    batch_size: int = 1024,  # Further increased batch size
    buffer_size: int = 2000000,  # Increased buffer size
    validation_episodes: int = 200,  # Increased validation episodes
    min_episodes: int = 20000,  # Increased minimum episodes
    max_episodes_without_improvement: int = 10000,  # Increased patience
    validation_threshold: float = 180.0,
    seed: int = 0,
    improvement_threshold: float = 1.0005,  # Reduced improvement threshold
    max_training_time: int = 864000,  # 24 hours
    warmup_steps: int = 20000,  # Increased warmup steps
    grad_clip_value: float = 0.5,  # Reduced gradient clipping
    lr_decay_steps: int = 100000,  # Increased lr decay steps
    adaptive_epsilon: bool = True,
    adaptive_lr: bool = True,
    double_dqn: bool = True,
    prioritized_replay: bool = True,
    epsilon_annealing_fraction: float = 0.7,  # Increased epsilon annealing fraction
    reward_scaling: float = 0.1,  # New parameter for reward scaling
    curriculum_learning: bool = True,  # Enable curriculum learning
    curriculum_start_difficulty: float = 0.1,  # Initial difficulty
    curriculum_end_difficulty: float = 1.0,  # Final difficulty
    curriculum_adaptation_rate: float = 0.01,  # Rate of difficulty increase
    num_levels: int = 2,  # Number of hierarchical levels
    sub_action_dims: List[int] = None  # Action dimensions for each level
) -> Tuple[List[train_state.TrainState], List[float], Dict[str, Any]]:
    rng = jax.random.PRNGKey(seed)
    logging.info(f"Initializing hierarchical RL agent training with seed {seed}")
    start_time = time.time()

    if sub_action_dims is None:
        sub_action_dims = [env.action_space.n] * num_levels
    assert len(sub_action_dims) == num_levels, "sub_action_dims must match num_levels"

    # Initialize the hierarchical agent and its state
    hierarchical_agent = HierarchicalRLAgent(features=agent.features, action_dims=sub_action_dims, num_levels=num_levels)
    hierarchical_state = create_train_state(rng, hierarchical_agent, env.reset()[0])
    target_hierarchical_state = create_train_state(jax.random.fold_in(rng, 1), hierarchical_agent, env.reset()[0])

    # Initialize variables to store sub-agent states
    sub_agent_states = hierarchical_state.params
    target_sub_agent_states = target_hierarchical_state.params

    # Initialize curriculum learning variables
    current_difficulty = curriculum_start_difficulty if curriculum_learning else curriculum_end_difficulty

    def adjust_difficulty(current_difficulty: float, performance: float) -> float:
        if curriculum_learning:
            target_performance = 0.7  # Adjust difficulty when performance is above 70%
            if performance > target_performance:
                current_difficulty = min(current_difficulty + curriculum_adaptation_rate, curriculum_end_difficulty)
            return current_difficulty
        return curriculum_end_difficulty

    try:
        dummy_obs, _ = env.reset()
        dummy_obs = jnp.array(dummy_obs)
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate / 10,
            peak_value=learning_rate,
            warmup_steps=num_episodes // 5,
            decay_steps=num_episodes,
            end_value=learning_rate / 1000
        )
        tx = optax.chain(
            optax.clip_by_global_norm(grad_clip_value),
            optax.adam(learning_rate=lr_schedule),
            optax.scale_by_schedule(optax.cosine_decay_schedule(1.0, num_episodes, 0.1))
        )
        state = create_train_state(rng, agent, dummy_obs, tx)
        target_state = create_train_state(rng, agent, dummy_obs, tx)
        replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=0.6, beta=0.4, beta_increment=0.001)
        logging.info(f"Initialized agent, target network, and prioritized replay buffer with size {buffer_size}")
        logging.info(f"Agent architecture: {agent}")
        logging.info(f"Training parameters: LR={learning_rate}, Gamma={gamma}, Epsilon={epsilon_start}->{epsilon_end}")
        logging.info(f"Episodes={num_episodes}, Max steps={max_steps}, Batch size={batch_size}")
        logging.info(f"Learning rate schedule: Warmup={num_episodes // 5}, Decay steps={num_episodes}")
        logging.info(f"Gradient clipping value: {grad_clip_value}")
    except Exception as e:
        logging.error(f"Error initializing RL agent: {str(e)}")
        raise RuntimeError(f"Failed to initialize RL agent: {str(e)}")

    @jax.jit
    def update_step(state: train_state.TrainState, target_state: train_state.TrainState, batch: Dict[str, jnp.ndarray], weights: jnp.ndarray):
        def loss_fn(params):
            q_values = state.apply_fn({'params': params}, batch['observations'])
            next_q_values = target_state.apply_fn({'params': target_state.params}, batch['next_observations'])
            next_actions = jnp.argmax(state.apply_fn({'params': params}, batch['next_observations']), axis=1)
            targets = batch['rewards'] + gamma * next_q_values[jnp.arange(batch_size), next_actions] * (1 - batch['dones'])
            td_errors = q_values[jnp.arange(batch_size), batch['actions']] - targets
            losses = optax.huber_loss(q_values[jnp.arange(batch_size), batch['actions']], targets, delta=1.0)
            loss = jnp.mean(losses * weights)
            return loss, td_errors

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, td_errors), grads = grad_fn(state.params)
        grads = jax.tree_map(lambda g: jnp.clip(g, -1.0, 1.0), grads)  # Gradient clipping
        updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
        new_params = optax.apply_updates(state.params, updates)
        return state.replace(params=new_params, opt_state=new_opt_state), loss, td_errors, jnp.mean(jnp.abs(td_errors))

    episode_rewards, shaped_rewards, recent_losses = [], [], []
    epsilon, best_average_reward = epsilon_start, float('-inf')
    episodes_without_improvement, total_steps = 0, 0
    solved, errors = False, []
    training_info = {
        'lr_history': [], 'epsilon_history': [], 'episode_lengths': [],
        'loss_history': [], 'shaped_rewards': [], 'reward_history': [], 'q_values': [],
        'grad_norms': [], 'param_norms': [], 'td_errors': [],
        'avg_q_values': [], 'max_q_values': [], 'min_q_values': [],
        'action_distributions': [], 'reward_per_action': {},
        'cumulative_reward': [], 'moving_avg_reward': [],
        'exploration_rate': [], 'exploitation_rate': [],
        'level_rewards': [[] for _ in range(num_levels)],
        'level_losses': [[] for _ in range(num_levels)]
    }
    adaptive_lr = learning_rate * 1.5  # Increased initial learning rate
    adaptive_epsilon = epsilon_start
    epsilon_decay_rate = (epsilon_end / epsilon_start) ** (1 / (num_episodes * 0.8))  # Slower decay

    logging.info(f"Starting training for {num_episodes} episodes, max {max_steps} steps per episode")
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            obs = jnp.array(obs)
            episode_reward = episode_shaped_reward = 0
            episode_losses, episode_q_values = [], []

            try:
                for step in range(max_steps):
                    total_steps += 1
                    # Implement a more sophisticated epsilon decay strategy for each level
                    epsilons = [max(epsilon_end, epsilon_start * (1 - (total_steps / (num_episodes * max_steps)) ** 1.5)) for _ in range(num_levels)]

                    # Adaptive epsilon strategy for each level
                    if adaptive_epsilon:
                        for level in range(num_levels):
                            if episode_reward > best_average_reward:
                                epsilons[level] *= 0.95  # Decrease epsilon faster if performing well
                            elif episode_reward < best_average_reward * 0.8:
                                epsilons[level] = min(epsilons[level] * 1.05, epsilon_start)  # Increase epsilon if performing poorly

                    rng, action_rng = jax.random.split(rng)
                    actions = []
                    q_values_list = []

                    # Hierarchical action selection
                    for level in range(num_levels):
                        q_values = sub_agent_states[level].apply_fn({'params': sub_agent_states[level].params}, obs[None, ...])
                        q_values_list.append(q_values)
                        episode_q_values.append(float(jnp.max(q_values)))

                        if jax.random.uniform(action_rng) < epsilons[level]:
                            action = jax.random.randint(action_rng, (), 0, agent.action_dims[level])
                            logging.debug(f"Episode {episode + 1}, Step {step + 1}, Level {level}: Random action {action}")
                        else:
                            action = int(jnp.argmax(q_values))
                            logging.debug(f"Episode {episode + 1}, Step {step + 1}, Level {level}: Greedy action {action}, Q-value: {float(jnp.max(q_values)):.4f}")

                        actions.append(action)

                        # Update observation for next level based on current level's action
                        if level < num_levels - 1:
                            obs = jnp.concatenate([obs, jax.nn.one_hot(action, agent.action_dims[level])])

                    # Apply curriculum learning by adjusting the environment difficulty
                    if curriculum_learning:
                        env.set_difficulty(current_difficulty)

                    next_obs, reward, done, truncated, _ = env.step(actions[-1])  # Use the action from the lowest level
                    next_obs = jnp.array(next_obs)

                    # Improve reward shaping mechanism with curriculum learning and hierarchical structure
                    shaped_reward = reward
                    for level in range(num_levels):
                        level_reward = 0.2 * (1 - abs(next_obs[level] / (env.observation_space.high[level] * current_difficulty)))
                        shaped_reward += level_reward * (num_levels - level)  # Higher levels have more impact

                    # Scale shaped reward based on current difficulty
                    shaped_reward *= current_difficulty

                    # Clip shaped reward to prevent extreme values
                    shaped_reward = jnp.clip(shaped_reward, -1.0, 3.0)
                    episode_shaped_reward += shaped_reward

                    # Add experiences to replay buffer for each level
                    for level in range(num_levels):
                        level_obs = obs[:sum(agent.action_dims[:level])]
                        level_next_obs = next_obs[:sum(agent.action_dims[:level])]
                        replay_buffer.add(level_obs, actions[level], shaped_reward, level_next_obs, done or truncated)

                    # Add intrinsic reward for exploration at each level
                    intrinsic_reward = sum(0.1 * (1 / (1 + episode_q_values.count(float(jnp.max(q))))) for q in q_values_list)
                    shaped_reward += intrinsic_reward

                    if len(replay_buffer) >= batch_size:
                        try:
                            for level in range(num_levels):
                                batch, indices, weights = replay_buffer.sample(batch_size)
                                sub_agent_states[level], loss, td_errors, _ = update_step(
                                    sub_agent_states[level], target_sub_agent_states[level],
                                    jax.tree_map(jnp.array, batch), weights
                                )
                                if jnp.isnan(loss) or jnp.isinf(loss):
                                    raise ValueError(f"NaN or Inf loss detected at level {level}: {loss}")
                                replay_buffer.update_priorities(indices, np.abs(td_errors))
                                episode_losses.append(float(loss))
                                logging.debug(f"Episode {episode + 1}, Step {step + 1}, Level {level}: Loss: {float(loss):.4f}")
                        except Exception as e:
                            logging.error(f"Error during update step: {str(e)}")
                            errors.append(f"Update step error: {str(e)}")
                            for level in range(num_levels):
                                sub_agent_states[level] = sub_agent_states[level].replace(opt_state=sub_agent_states[level].tx.init(sub_agent_states[level].params))
                            epsilons = [min(eps * 1.05, 1.0) for eps in epsilons]
                            logging.warning(f"Resetting optimizer states and increasing epsilons to {epsilons}")

                    if total_steps % target_update_freq == 0:
                        for level in range(num_levels):
                            target_sub_agent_states[level] = target_sub_agent_states[level].replace(params=sub_agent_states[level].params)
                        logging.debug(f"Updated target networks at step {total_steps}")

                    episode_reward += reward
                    obs = next_obs
                    if done or truncated:
                        logging.info(f"Episode {episode + 1} ended after {step + 1} steps with reward {episode_reward:.2f}")
                        break

                # Adjust difficulty based on episode performance
                if curriculum_learning:
                    current_difficulty = adjust_difficulty(current_difficulty, episode_reward)
                    logging.info(f"Adjusted difficulty to {current_difficulty:.2f}")

                episode_rewards.append(episode_reward)
                shaped_rewards.append(episode_shaped_reward)
                current_lr = lr_schedule(episode)

                logging.info(f"Episode {episode + 1} completed: Reward = {episode_reward:.2f}, Shaped Reward = {episode_shaped_reward:.2f}")

            except Exception as e:
                logging.error(f"Error during episode {episode + 1}: {str(e)}")
                errors.append(f"Episode error: {str(e)}")
                continue  # Move to the next episode

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

            if len(recent_losses) >= 100:
                window_size = 100
                loss_ratios = [max(recent_losses[i:i+window_size]) / (min(recent_losses[i:i+window_size]) + 1e-8) for i in range(len(recent_losses) - window_size + 1)]
                avg_loss_ratio = np.mean(loss_ratios[-20:])
                moving_avg_loss = np.mean(recent_losses[-50:])
                baseline_loss = np.mean(recent_losses[:50])

                if avg_loss_ratio > 10 or jnp.isnan(jnp.array(recent_losses[-100:])).any() or moving_avg_loss > 2 * baseline_loss:
                    logging.warning(f"Detected significant training instability. Avg loss ratio: {avg_loss_ratio:.2f}, Moving avg loss: {moving_avg_loss:.4f}")
                    state = state.replace(opt_state=state.tx.init(state.params))
                    epsilon = min(epsilon * 1.1, 0.3)
                    learning_rate *= 0.8
                    tx = optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adam(learning_rate=learning_rate)
                    )
                    state = state.replace(tx=tx)
                    episodes_without_improvement = 0
                    logging.info(f"Reset optimizer state, adjusted epsilon to {epsilon:.4f}, and reduced learning rate to {learning_rate:.6f}")
                elif avg_loss_ratio > 3:
                    logging.info(f"Minor instability detected. Adjusting learning rate.")
                    learning_rate *= 0.95
                    tx = optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adam(learning_rate=learning_rate)
                    )
                    state = state.replace(tx=tx)

                # Add a mechanism to increase learning rate if progress is too slow
                if episodes_without_improvement > 50:
                    learning_rate *= 1.05
                    tx = optax.chain(
                        optax.clip_by_global_norm(1.0),
                        optax.adam(learning_rate=learning_rate)
                    )
                    state = state.replace(tx=tx)
                    logging.info(f"Increased learning rate to {learning_rate:.6f} due to lack of improvement")

            if avg_shaped_reward >= early_stop_threshold and episode >= min_episodes:
                validation_rewards = run_validation(state, env, validation_episodes, max_steps)
                val_avg_reward = sum(validation_rewards) / validation_episodes
                val_std_reward = np.std(validation_rewards)
                if val_avg_reward >= validation_threshold and val_std_reward < early_stop_threshold * 0.2:
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
                    epsilon = min(epsilon * 1.1, 1.0)
                    current_lr *= 0.8
                    state = state.replace(tx=optax.chain(
                        optax.clip_by_global_norm(0.5),
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

    # Generate visualizations
    generate_learning_curve(episode_rewards, training_info['lr_history'], training_info['epsilon_history'])
    generate_q_value_distribution(q_values_list)
    generate_epsilon_decay(training_info['epsilon_history'])

    return state, episode_rewards, training_info

def generate_learning_curve(rewards, lr_history, epsilon_history):
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.plot(rewards)
    plt.title('Learning Curve')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(3, 1, 2)
    plt.plot(lr_history)
    plt.title('Learning Rate')
    plt.xlabel('Episode')
    plt.ylabel('Learning Rate')

    plt.subplot(3, 1, 3)
    plt.plot(epsilon_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')

    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.close()

def generate_q_value_distribution(q_values_list):
    plt.figure(figsize=(10, 6))
    plt.hist(q_values_list, bins=50)
    plt.title('Q-value Distribution')
    plt.xlabel('Q-value')
    plt.ylabel('Frequency')
    plt.savefig('q_value_distribution.png')
    plt.close()

def generate_epsilon_decay(epsilon_history):
    plt.figure(figsize=(10, 6))
    plt.plot(epsilon_history)
    plt.title('Epsilon Decay')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    plt.savefig('epsilon_decay.png')
    plt.close()

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
