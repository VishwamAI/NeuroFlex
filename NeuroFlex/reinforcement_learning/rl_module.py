# MIT License
# 
# Copyright (c) 2024 VishwamAI
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Any, Sequence, Callable
import gym
from gym import spaces
import logging
import numpy as np
import time
import scipy.signal
from ..utils import utils

# Constants for PPO
gamma = 0.99
lam = 0.95
value_loss_coef = 0.5
entropy_coef = 0.01

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = torch.zeros((size, *obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((size, *act_dim), dtype=torch.float32)
        self.adv_buf = torch.zeros(size, dtype=torch.float32)
        self.rew_buf = torch.zeros(size, dtype=torch.float32)
        self.ret_buf = torch.zeros(size, dtype=torch.float32)
        self.val_buf = torch.zeros(size, dtype=torch.float32)
        self.logp_buf = torch.zeros(size, dtype=torch.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

class PrioritizedReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.obs_buf = torch.zeros((capacity, *obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((capacity, *act_dim), dtype=torch.float32)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32)
        self.val_buf = torch.zeros(capacity, dtype=torch.float32)
        self.ret_buf = torch.zeros(capacity, dtype=torch.float32)
        self.adv_buf = torch.zeros(capacity, dtype=torch.float32)
        self.logp_buf = torch.zeros(capacity, dtype=torch.float32)
        self.priorities = torch.zeros(capacity, dtype=torch.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, capacity

    def add(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = torch.as_tensor(obs)
        self.act_buf[self.ptr] = torch.as_tensor(act)
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = torch.cat([self.rew_buf[path_slice], torch.tensor([last_val])])
        vals = torch.cat([self.val_buf[path_slice], torch.tensor([last_val])])

        # GAE-Lambda advantage calculation
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, gamma * lam)

        # Rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        # Advantage normalization
        adv_mean, adv_std = self.adv_buf.mean(), self.adv_buf.std()
        self.adv_buf = (self.adv_buf - adv_mean) / (adv_std + 1e-8)
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: v for k, v in data.items()}

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

def compute_gae(state, buffer, gamma, lam=0.95):
    values = state.apply_fn({'params': state.params}, buffer.obs_buf)
    last_gae_lam = 0
    for step in reversed(range(buffer.ptr)):
        if step == buffer.ptr - 1:
            next_non_terminal = 1.0
            next_values = 0  # Assuming 0 for terminal state
        else:
            next_non_terminal = 1.0
            next_values = values[step + 1]
        delta = buffer.rew_buf[step] + gamma * next_values * next_non_terminal - values[step]
        buffer.adv_buf[step] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
    buffer.ret_buf[:buffer.ptr] = buffer.adv_buf[:buffer.ptr] + values[:buffer.ptr]
    return buffer.adv_buf[:buffer.ptr]

class Actor(nn.Module):
    def __init__(self, action_dim: int, features: List[int]):
        super().__init__()
        self.action_dim = action_dim
        self.features = features
        self.layers = nn.ModuleList()
        for i, feat in enumerate(self.features):
            self.layers.append(nn.Linear(features[i-1] if i > 0 else features[0], feat))
            self.layers.append(nn.LayerNorm(feat))
            self.layers.append(nn.ReLU())
        self.policy_logits = nn.Linear(features[-1], self.action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        logits = self.policy_logits(x)
        return logits

class Critic(nn.Module):
    def __init__(self, features: List[int]):
        super(Critic, self).__init__()
        self.features = features
        self.layers = nn.ModuleList()
        for i, feat in enumerate(self.features):
            self.layers.append(nn.Linear(features[i-1] if i > 0 else features[0], feat))
            self.layers.append(nn.LayerNorm(feat))
            self.layers.append(nn.ReLU())
        self.value = nn.Linear(features[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.value(x)

class RLAgent(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, features: List[int] = [256, 256, 256]):
        super(RLAgent, self).__init__()
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.features = features
        logging.info(f"Setting up RLAgent with observation_dim: {self.observation_dim}, action_dim: {self.action_dim}, features: {self.features}")
        self.actor = Actor(self.action_dim, [self.observation_dim] + self.features)
        self.critic = Critic([self.observation_dim] + self.features)
        logging.info(f"RLAgent setup complete. Actor: {self.actor}, Critic: {self.critic}")

    def forward(self, x):
        logging.debug(f"RLAgent forward pass with input shape: {x.shape}")
        return self.actor(x), self.critic(x)

    def actor_forward(self, x):
        logging.debug(f"Actor forward pass with input shape: {x.shape}")
        return self.actor(x)

    def critic_forward(self, x):
        logging.debug(f"Critic forward pass with input shape: {x.shape}")
        return self.critic(x)

    def initialize_params(self, input_shape):
        logging.info(f"Initializing RLAgent parameters with input shape: {input_shape}")
        dummy_input = torch.ones((1,) + input_shape)
        self(dummy_input)  # Forward pass to initialize parameters
        self._check_params_structure()
        logging.info(f"RLAgent parameters initialized.")
        logging.debug(f"Actor params structure: {self.actor.state_dict().keys()}")
        logging.debug(f"Critic params structure: {self.critic.state_dict().keys()}")

    def _check_params_structure(self):
        for submodule in ['actor', 'critic']:
            if not hasattr(self, submodule):
                logging.error(f"{submodule} not found in RLAgent")
                raise ValueError(f"RLAgent initialization did not create {submodule} submodule")
            if not isinstance(getattr(self, submodule), nn.Module):
                logging.error(f"{submodule} is not a nn.Module. Type: {type(getattr(self, submodule))}")
                raise ValueError(f"{submodule} initialization did not return a nn.Module")
            if not any(name.startswith('0') for name in getattr(self, submodule).state_dict().keys()):
                logging.error(f"First layer not found in {submodule} params. Keys: {getattr(self, submodule).state_dict().keys()}")
                raise ValueError(f"{submodule} initialization did not create expected layer structure")

    @property
    def actor_params(self):
        return self.actor.state_dict()

    @property
    def critic_params(self):
        return self.critic.state_dict()

class RLEnvironment:
    def __init__(self, env_name: str, seed: int = 42):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = seed
        self.env.reset(seed=self.seed)
        self.episode_step = 0
        self.max_episode_steps = self.env._max_episode_steps  # Get max steps from env

    def reset(self) -> Tuple[torch.Tensor, Dict]:
        observation, info = self.env.reset(seed=self.seed)
        self.episode_step = 0
        logging.info(f"Environment reset. Initial observation: {observation}")
        return torch.tensor(observation, dtype=torch.float32), info

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        done = terminated or truncated or (self.episode_step >= self.max_episode_steps)
        logging.debug(f"Step {self.episode_step}: Action={action}, Reward={reward}, Done={done}")
        return torch.tensor(obs, dtype=torch.float32), reward, done, truncated, info

    def is_episode_done(self, done: bool, truncated: bool) -> bool:
        episode_done = done or truncated or (self.episode_step >= self.max_episode_steps)
        if episode_done:
            logging.info(f"Episode completed after {self.episode_step} steps.")
        return episode_done

def create_train_state(model, dummy_input, optimizer):
    if isinstance(dummy_input, tuple):
        dummy_input = dummy_input[0]  # Extract observation from tuple
    dummy_input = torch.tensor(dummy_input, dtype=torch.float32)  # Ensure input is a PyTorch tensor
    logging.info(f"Creating train state with dummy input shape: {dummy_input.shape}")

    try:
        model.to(dummy_input.device)
        _ = model(dummy_input.unsqueeze(0))  # Forward pass to initialize parameters
        logging.info("Model initialization successful.")
        logging.debug(f"Full model structure: {model}")
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        logging.error(f"Model structure: {model}")
        raise ValueError(f"Failed to initialize model: {str(e)}")

    params = {name: param for name, param in model.named_parameters()}
    logging.info(f"Initial params structure: {[(name, param.shape) for name, param in params.items()]}")

    if not isinstance(params, dict):
        logging.error(f"Params is not a dictionary. Type: {type(params)}")
        raise ValueError("Model initialization did not return a dictionary for params")

    if not hasattr(model, 'actor') or not hasattr(model, 'critic'):
        logging.error("Missing 'actor' or 'critic' in model.")
        raise ValueError("Model does not have 'actor' and 'critic' submodules")

    logging.info(f"Actor params structure: {[(name, param.shape) for name, param in model.actor.named_parameters()]}")
    logging.info(f"Critic params structure: {[(name, param.shape) for name, param in model.critic.named_parameters()]}")

    # Add assertions to ensure 'actor' and 'critic' are properly structured
    assert isinstance(model.actor, torch.nn.Module), f"'actor' should be a torch.nn.Module, got {type(model.actor)}"
    assert isinstance(model.critic, torch.nn.Module), f"'critic' should be a torch.nn.Module, got {type(model.critic)}"

    try:
        # Create separate optimizers for actor and critic
        optimizer_actor = optimizer(model.actor.parameters())
        optimizer_critic = optimizer(model.critic.parameters())
        logging.info(f"Optimizer structure - Actor: {optimizer_actor}, Critic: {optimizer_critic}")

        state = {
            'model': model,
            'optimizer': {'actor': optimizer_actor, 'critic': optimizer_critic}
        }
        logging.info("Train state created successfully")
        logging.debug(f"Train state structure: {state}")
    except Exception as e:
        logging.error(f"Error creating train state: {str(e)}")
        logging.error(f"Model structure: {model}")
        logging.error(f"Optimizer structure: {optimizer}")
        raise ValueError(f"Failed to create train state: {str(e)}")

    logging.info(f"Created train state with model structure: {model}")
    logging.debug(f"Actor params in state: {[(name, param.shape) for name, param in model.actor.named_parameters()]}")
    logging.debug(f"Critic params in state: {[(name, param.shape) for name, param in model.critic.named_parameters()]}")
    logging.debug(f"Full state structure: {state}")

    logging.info("Train state creation successful with proper 'actor' and 'critic' submodules")

    return state

def select_action(model: nn.Module, observation: torch.Tensor, epsilon: float = 0.0, training: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    logging.info(f"Selecting action for observation shape: {observation.shape}")

    try:
        logging.debug(f"Applying actor forward pass with observation shape: {observation.unsqueeze(0).shape}")

        with torch.no_grad():
            logits = model.actor(observation.unsqueeze(0))
        logging.debug(f"Action logits shape: {logits.shape}, content: {logits}")

        if training and torch.rand(1).item() < epsilon:
            action = torch.randint(0, logits.shape[-1], (1,))
        else:
            action = torch.argmax(logits, dim=-1)
        log_prob = torch.nn.functional.log_softmax(logits, dim=-1)[0, action]
        logging.debug(f"Selected action: {action}, Log probability: {log_prob}")

        return action, log_prob
    except Exception as e:
        logging.error(f"Actor forward pass failed: {str(e)}")
        logging.error(f"Model: {model}")
        logging.error(f"Observation: {observation}")
        raise RuntimeError(f"Actor forward pass failed: {str(e)}") from e

from typing import Dict, Generator, Any

def get_minibatches(data: Dict[str, torch.Tensor], batch_size: int) -> Generator[Dict[str, torch.Tensor], None, None]:
    data_size = len(data['obs'])
    assert all(len(v) == data_size for v in data.values()), "All tensors must have the same length"
    indices = torch.randperm(data_size)
    start_idx = 0
    while start_idx < data_size:
        end_idx = min(start_idx + batch_size, data_size)
        batch_indices = indices[start_idx:end_idx]
        yield {k: v[batch_indices] for k, v in data.items()}
        start_idx += batch_size

def train_rl_agent(
    agent: RLAgent,
    env: RLEnvironment,
    num_episodes: int = 10000,
    max_steps: int = 1000,
    gamma: float = 0.99,
    clip_ratio: float = 0.2,
    learning_rate: float = 3e-4,
    n_epochs: int = 10,
    batch_size: int = 64,
    buffer_size: int = 2048,
    target_kl: float = 0.01,
    value_loss_coef: float = 0.5,
    entropy_coef: float = 0.01,
    early_stop_threshold: float = 195.0,
    early_stop_episodes: int = 100,
    validation_episodes: int = 20,
    min_episodes: int = 2000,
    max_episodes_without_improvement: int = 1000,
    validation_threshold: float = 180.0,
    seed: int = 0,
    improvement_threshold: float = 1.005,
    max_training_time: int = 72000,
) -> Tuple[RLAgent, List[float], Dict[str, Any]]:
    torch.manual_seed(seed)
    logging.info(f"Initializing PPO agent training with seed {seed}")
    start_time = time.time()

    try:
        dummy_obs, _ = env.reset()
        dummy_obs = torch.tensor(dummy_obs, dtype=torch.float32)
        optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
        buffer = PPOBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)
        logging.info(f"Initialized agent and PPO buffer with size {buffer_size}")
        logging.info(f"Agent architecture: {agent}")
        logging.info(f"Training parameters: LR={learning_rate}, Gamma={gamma}, Clip ratio={clip_ratio}")
        logging.info(f"Episodes={num_episodes}, Max steps={max_steps}, Batch size={batch_size}")
    except Exception as e:
        logging.error(f"Error initializing PPO agent: {str(e)}")
        raise RuntimeError(f"Failed to initialize PPO agent: {str(e)}")

    def update_ppo(agent: RLAgent, batch: Dict[str, torch.Tensor]):
        pi, v = agent(batch['obs'])
        log_prob = torch.nn.functional.log_softmax(pi, dim=-1).gather(1, batch['act'].unsqueeze(-1)).squeeze(-1)
        ratio = torch.exp(log_prob - batch['logp'])
        clip_adv = torch.clamp(ratio, 1-clip_ratio, 1+clip_ratio) * batch['adv']
        policy_loss = -torch.mean(torch.min(ratio * batch['adv'], clip_adv))

        value_pred = v.squeeze(-1)
        value_loss = 0.5 * torch.mean((value_pred - batch['ret'])**2)

        entropy = -torch.mean(torch.sum(torch.nn.functional.softmax(pi, dim=-1) * torch.nn.functional.log_softmax(pi, dim=-1), dim=-1))
        kl = torch.mean(batch['logp'] - log_prob)

        total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        return policy_loss.item(), value_loss.item(), kl.item(), entropy.item()

    episode_rewards = []
    best_average_reward = float('-inf')
    episodes_without_improvement = 0
    solved = False
    errors = []
    training_info = {'policy_loss_history': [], 'value_loss_history': [], 'kl_history': [],
                     'episode_lengths': [], 'reward_history': []}

    logging.info(f"Starting training for {num_episodes} episodes, max {max_steps} steps per episode")
    try:
        for episode in range(num_episodes):
            obs, _ = env.reset()
            episode_reward = 0
            episode_length = 0

            for step in range(max_steps):
                action, log_prob = agent.select_action(torch.tensor(obs, dtype=torch.float32))
                next_obs, reward, done, truncated, _ = env.step(action.item())

                episode_reward += reward
                episode_length += 1

                value = agent.critic(torch.tensor(obs, dtype=torch.float32).unsqueeze(0)).squeeze()
                buffer.add(obs, action, reward, value, log_prob)

                if buffer.ptr == buffer.max_size:
                    last_val = agent.critic(torch.tensor(next_obs, dtype=torch.float32).unsqueeze(0)).squeeze()
                    buffer.finish_path(last_val)
                    data = buffer.get()
                    for _ in range(n_epochs):
                        for mini_batch in get_minibatches(data, batch_size):
                            policy_loss, value_loss, kl, entropy = update_ppo(agent, mini_batch)
                            training_info['policy_loss_history'].append(policy_loss)
                            training_info['value_loss_history'].append(value_loss)
                            training_info['kl_history'].append(kl)
                            if kl > 1.5 * target_kl:
                                logging.info(f"Early stopping at step {step}, epoch {_} due to reaching max KL.")
                                break
                    buffer = PPOBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)

                obs = next_obs
                if done or truncated:
                    break

            episode_rewards.append(episode_reward)
            avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)

            logging.info(f"Episode {episode + 1}/{num_episodes}, Steps: {episode_length}, "
                         f"Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

            training_info['reward_history'].append(episode_reward)
            training_info['episode_lengths'].append(episode_length)

            if avg_reward > best_average_reward * improvement_threshold:
                best_average_reward = avg_reward
                episodes_without_improvement = 0
            else:
                episodes_without_improvement += 1

            # Early stopping checks
            if avg_reward >= early_stop_threshold and episode >= min_episodes:
                validation_rewards = run_validation(agent, env, validation_episodes, max_steps)
                val_avg_reward = sum(validation_rewards) / validation_episodes
                val_std_reward = torch.tensor(validation_rewards).std().item()
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

            if time.time() - start_time > max_training_time:
                logging.warning(f"Maximum training time of {max_training_time} seconds reached. Stopping.")
                training_info['early_stop_reason'] = 'max_training_time_reached'
                break

    except Exception as e:
        logging.error(f"Unexpected error during training: {str(e)}")
        logging.error(f"Current episode: {episode}")
        errors.append(f"Training error: {str(e)}")
        training_info['training_stopped_early'] = True

    if not episode_rewards:
        raise ValueError("No episodes completed successfully")

    logging_message = "Training completed successfully" if solved else "Training completed without solving"
    logging.info(f"{logging_message}. Best average reward: {best_average_reward:.2f}")

    training_info.update({
        'best_average_reward': best_average_reward,
        'total_episodes': episode + 1,
        'solved': solved,
        'errors': errors,
        'validation_rewards': validation_rewards if 'validation_rewards' in locals() else None,
        'training_stopped_early': episodes_without_improvement >= max_episodes_without_improvement,
        'total_training_time': time.time() - start_time
    })

    logging.info("Training completed. Returning results.")
    return agent, episode_rewards, training_info

def run_validation(agent: RLAgent, env: RLEnvironment, num_episodes: int, max_steps: int) -> List[float]:
    validation_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action, _ = agent.select_action(torch.tensor(obs, dtype=torch.float32))
            obs, reward, done, truncated, _ = env.step(action.item())
            episode_reward += reward
            if done or truncated:
                break
        validation_rewards.append(episode_reward)
    return validation_rewards

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: Tuple[int, ...], act_dim: Tuple[int, ...], batch_size: int = 32, alpha: float = 0.6):
        self.capacity = capacity
        self.batch_size = batch_size
        self.alpha = alpha
        self.obs_buf = torch.zeros((capacity, *obs_dim), dtype=torch.float32)
        self.act_buf = torch.zeros((capacity, *act_dim), dtype=torch.long)
        self.rew_buf = torch.zeros(capacity, dtype=torch.float32)
        self.next_obs_buf = torch.zeros((capacity, *obs_dim), dtype=torch.float32)
        self.done_buf = torch.zeros(capacity, dtype=torch.bool)
        self.priorities = torch.zeros(capacity, dtype=torch.float32)
        self.position = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done):
        max_priority = self.priorities.max().item() if self.size > 0 else 1.0
        self.obs_buf[self.position] = torch.as_tensor(obs)
        self.act_buf[self.position] = torch.as_tensor(action)
        self.rew_buf[self.position] = reward
        self.next_obs_buf[self.position] = torch.as_tensor(next_obs)
        self.done_buf[self.position] = done
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int = None, beta: float = 0.4):
        if self.size == 0:
            return None

        batch_size = batch_size or self.batch_size
        priorities = self.priorities[:self.size]
        probabilities = priorities.pow(self.alpha)
        probabilities /= probabilities.sum()

        indices = torch.multinomial(probabilities, batch_size, replacement=True)
        weights = (self.size * probabilities[indices]).pow(-beta)
        weights /= weights.max()

        return {
            'observations': self.obs_buf[indices],
            'actions': self.act_buf[indices],
            'rewards': self.rew_buf[indices],
            'next_observations': self.next_obs_buf[indices],
            'dones': self.done_buf[indices],
            'indices': indices,
            'weights': weights
        }

    def update_priorities(self, indices, priorities):
        self.priorities[indices] = priorities

    def __len__(self):
        return self.size
