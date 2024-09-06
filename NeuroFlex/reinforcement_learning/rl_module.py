import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Tuple, Dict, Any, Sequence, Callable, NamedTuple
import gym
from gym import spaces
import optax
import logging
import numpy as np
import time
import dataclasses
from .extended_train_state import ExtendedTrainState
import scipy.signal
import random
from ..utils import utils

# Constants for PPO
gamma = 0.99
lam = 0.95
value_loss_coef = 0.5
entropy_coef = 0.01

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PPOBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.obs_buf = np.zeros((size, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, *act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0

    def add(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

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
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: jnp.array(v) for k,v in data.items()}

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
    action_dim: int
    features: List[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f'Dense_{i}', kernel_init=nn.initializers.he_normal())(x)
            x = nn.LayerNorm(name=f'LayerNorm_{i}')(x)
            x = nn.relu(x)

        logits = nn.Dense(self.action_dim, name='policy_logits', kernel_init=nn.initializers.he_normal())(x)
        return logits

class Critic(nn.Module):
    features: List[int]

    @nn.compact
    def __call__(self, x):
        for i, feat in enumerate(self.features):
            x = nn.Dense(
                feat,
                name=f'Dense_{i}',
                kernel_init=nn.initializers.he_normal()
            )(x)
            x = nn.LayerNorm(name=f'LayerNorm_{i}')(x)
            x = nn.relu(x)

        value = nn.Dense(
            1,
            name='value',
            kernel_init=nn.initializers.he_normal()
        )(x)
        return value

class RLAgent(nn.Module):
    action_dim: int
    features: List[int] = dataclasses.field(default_factory=lambda: [256, 256, 256])

    def setup(self):
        logging.info(f"Setting up RLAgent with action_dim: {self.action_dim}, features: {self.features}")
        self.actor = Actor(self.action_dim, self.features)
        self.critic = Critic(self.features)
        logging.info(f"RLAgent setup complete. Actor: {self.actor}, Critic: {self.critic}")
        logging.debug(f"Actor structure: {self.actor.tabulate(jax.random.PRNGKey(0), jnp.ones((1, self.features[0])))}")
        logging.debug(f"Critic structure: {self.critic.tabulate(jax.random.PRNGKey(0), jnp.ones((1, self.features[0])))}")

    def __call__(self, x):
        logging.debug(f"RLAgent __call__ with input shape: {x.shape}")
        return self.actor(x), self.critic(x)

    def actor_forward(self, x):
        logging.debug(f"Actor forward pass with input shape: {x.shape}")
        return self.actor(x)

    def critic_forward(self, x):
        logging.debug(f"Critic forward pass with input shape: {x.shape}")
        return self.critic(x)

    def initialize_params(self, rng, input_shape):
        logging.info(f"Initializing RLAgent parameters with input shape: {input_shape}")
        dummy_input = jnp.ones((1,) + input_shape)
        variables = self.init(rng, dummy_input)
        params = variables['params']
        self._check_params_structure(params)
        logging.info(f"RLAgent parameters initialized. Structure: {jax.tree_map(lambda x: x.shape, params)}")
        logging.debug(f"Actor params structure: {jax.tree_map(lambda x: x.shape, params['actor'])}")
        logging.debug(f"Critic params structure: {jax.tree_map(lambda x: x.shape, params['critic'])}")
        return params

    def _check_params_structure(self, params):
        if 'actor' not in params or 'critic' not in params:
            logging.error(f"Model initialization failed. Params keys: {params.keys()}")
            raise ValueError("Model initialization did not create 'actor' and 'critic' submodules")

        for submodule in ['actor', 'critic']:
            if not isinstance(params[submodule], dict):
                logging.error(f"{submodule} params is not a dictionary. Type: {type(params[submodule])}")
                raise ValueError(f"{submodule} initialization did not return a dictionary for params")
            if 'Dense_0' not in params[submodule]:
                logging.error(f"'Dense_0' not found in {submodule} params. Keys: {params[submodule].keys()}")
                raise ValueError(f"{submodule} initialization did not create expected layer structure")

    def _structure_params(self, params):
        logging.info("Structuring RLAgent parameters")
        if 'actor' not in params or 'critic' not in params:
            logging.error(f"Invalid params structure. Keys: {params.keys()}")
            raise ValueError("Params must contain 'actor' and 'critic' keys")

        structured_params = {
            'actor': params['actor'],
            'critic': params['critic']
        }
        logging.debug(f"Structured params: {jax.tree_map(lambda x: x.shape, structured_params)}")
        return structured_params

    @property
    def actor_params(self):
        if not hasattr(self, 'params') or 'actor' not in self.params:
            logging.error("Actor params not found in RLAgent")
            return None
        return self.params['actor']

    @property
    def critic_params(self):
        if not hasattr(self, 'params') or 'critic' not in self.params:
            logging.error("Critic params not found in RLAgent")
            return None
        return self.params['critic']

class RLEnvironment:
    def __init__(self, env_name: str, seed: int = 42):
        self.env = gym.make(env_name)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self.seed = seed
        self.env.reset(seed=self.seed)
        self.episode_step = 0
        self.max_episode_steps = self.env._max_episode_steps  # Get max steps from env

    def reset(self) -> Tuple[jnp.ndarray, Dict]:
        observation, info = self.env.reset(seed=self.seed)
        self.episode_step = 0
        logging.info(f"Environment reset. Initial observation: {observation}")
        return jnp.array(observation), info

    def step(self, action: int) -> Tuple[jnp.ndarray, float, bool, bool, Dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_step += 1
        done = terminated or truncated or (self.episode_step >= self.max_episode_steps)
        logging.debug(f"Step {self.episode_step}: Action={action}, Reward={reward}, Done={done}")
        return jnp.array(obs), reward, done, truncated, info

    def is_episode_done(self, done: bool, truncated: bool) -> bool:
        episode_done = done or truncated or (self.episode_step >= self.max_episode_steps)
        if episode_done:
            logging.info(f"Episode completed after {self.episode_step} steps.")
        return episode_done

def create_train_state(rng, model, dummy_input, tx):
    if isinstance(dummy_input, tuple):
        dummy_input = dummy_input[0]  # Extract observation from tuple
    dummy_input = jnp.array(dummy_input)  # Ensure input is a JAX array
    logging.info(f"Creating train state with dummy input shape: {dummy_input.shape}")

    try:
        variables = model.init(rng, dummy_input[None, ...])
        logging.info(f"Model initialization successful. Variables keys: {variables.keys()}")
        logging.debug(f"Full variables structure: {jax.tree_map(lambda x: x.shape, variables)}")
    except Exception as e:
        logging.error(f"Error initializing model: {str(e)}")
        logging.error(f"Model structure: {model}")
        logging.error(f"RNG key: {rng}")
        raise ValueError(f"Failed to initialize model: {str(e)}")

    if 'params' not in variables:
        logging.error(f"'params' not found in initialized variables. Keys: {variables.keys()}")
        raise ValueError("Model initialization did not return 'params'")

    params = variables['params']
    logging.info(f"Initial params structure: {jax.tree_map(lambda x: x.shape, params)}")
    logging.debug(f"Initial params content: {params}")

    params = model._structure_params(params)  # Use the _structure_params method to structure the params
    logging.info(f"Structured model parameters: {jax.tree_map(lambda x: x.shape, params)}")
    logging.debug(f"Structured params content: {params}")

    if not isinstance(params, dict):
        logging.error(f"Params is not a dictionary. Type: {type(params)}")
        raise ValueError("Model initialization did not return a dictionary for params")

    if 'actor' not in params or 'critic' not in params:
        logging.error(f"Missing 'actor' or 'critic' in params. Keys found: {params.keys()}")
        raise ValueError("Model initialization did not create 'actor' and 'critic' submodules")

    logging.info(f"Actor params structure: {jax.tree_map(lambda x: x.shape, params['actor'])}")
    logging.info(f"Critic params structure: {jax.tree_map(lambda x: x.shape, params['critic'])}")
    logging.debug(f"Actor params content: {params['actor']}")
    logging.debug(f"Critic params content: {params['critic']}")

    # Add assertions to ensure 'actor' and 'critic' are properly structured
    assert isinstance(params['actor'], dict), f"'actor' params should be a dictionary, got {type(params['actor'])}"
    assert isinstance(params['critic'], dict), f"'critic' params should be a dictionary, got {type(params['critic'])}"
    assert 'Dense_0' in params['actor'], f"'actor' params should contain 'Dense_0'. Keys: {params['actor'].keys()}"
    assert 'Dense_0' in params['critic'], f"'critic' params should contain 'Dense_0'. Keys: {params['critic'].keys()}"

    def apply_fn(params, x, method=None):
        logging.debug(f"apply_fn called with method: {method}")
        if method == 'actor_forward':
            actor_params = params['actor'] if 'actor' in params else params
            logging.debug(f"Actor forward pass with params: {jax.tree_map(lambda x: x.shape, actor_params)}")
            return model.apply(actor_params, x, method=model.actor_forward)
        elif method == 'critic_forward':
            critic_params = params['critic'] if 'critic' in params else params
            logging.debug(f"Critic forward pass with params: {jax.tree_map(lambda x: x.shape, critic_params)}")
            return model.apply(critic_params, x, method=model.critic_forward)
        else:
            logging.debug(f"Full model forward pass with params: {jax.tree_map(lambda x: x.shape, params)}")
            return model.apply(params, x)

    try:
        # Create separate optimizers for actor and critic
        tx_actor = tx.actor if isinstance(tx, dict) else tx
        tx_critic = tx.critic if isinstance(tx, dict) else tx
        logging.info(f"Optimizer structure - Actor: {tx_actor}, Critic: {tx_critic}")

        state = ExtendedTrainState.create(
            apply_fn=apply_fn,
            params=params,
            tx={'actor': tx_actor, 'critic': tx_critic},
            batch_stats=variables.get('batch_stats', {})
        )
        logging.info("ExtendedTrainState created successfully")
        logging.debug(f"ExtendedTrainState structure: {state}")
    except Exception as e:
        logging.error(f"Error creating ExtendedTrainState: {str(e)}")
        logging.error(f"Params structure: {jax.tree_map(lambda x: x.shape, params)}")
        logging.error(f"Optimizer structure: {tx}")
        raise ValueError(f"Failed to create ExtendedTrainState: {str(e)}")

    logging.info(f"Created ExtendedTrainState with params structure: {jax.tree_map(lambda x: x.shape, state.params)}")
    logging.debug(f"Actor params in state: {jax.tree_map(lambda x: x.shape, state.params['actor'])}")
    logging.debug(f"Critic params in state: {jax.tree_map(lambda x: x.shape, state.params['critic'])}")
    logging.debug(f"Full state structure: {state}")

    logging.info("ExtendedTrainState creation successful with proper 'actor' and 'critic' submodules")

    return state

def select_action(state: ExtendedTrainState, observation: jnp.ndarray, rng: jax.random.PRNGKey) -> Tuple[jnp.ndarray, jnp.ndarray]:
    logging.info(f"Selecting action for observation shape: {observation.shape}")
    logging.debug(f"State params keys: {state.params.keys()}")

    if not isinstance(state.params, dict):
        logging.error(f"Invalid state.params type: {type(state.params)}")
        raise TypeError("state.params must be a dictionary")

    # Correctly access actor params
    actor_params = state.params.get('actor', state.params)
    if not isinstance(actor_params, dict):
        logging.error(f"Invalid actor params type: {type(actor_params)}")
        raise TypeError("Actor params must be a dictionary")

    logging.debug(f"Actor params structure: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), actor_params)}")
    logging.debug(f"Actor params keys: {list(actor_params.keys())}")

    # Check for expected actor layers
    expected_layers = ['Dense_0', 'LayerNorm_0', 'Dense_1', 'LayerNorm_1', 'policy_logits']
    missing_layers = [layer for layer in expected_layers if layer not in actor_params]
    if missing_layers:
        logging.error(f"Expected layers not found in actor params: {missing_layers}")
        raise ValueError(f"Actor params missing expected layers: {missing_layers}")

    try:
        logging.debug(f"Applying actor forward pass with observation shape: {observation[None, ...].shape}")

        logits = state.apply_fn(
            {'params': {'actor': actor_params}},
            observation[None, ...],
            method='actor_forward'
        )
        logging.debug(f"Action logits shape: {logits.shape}, content: {logits}")

        action = jax.random.categorical(rng, logits[0])
        log_prob = jax.nn.log_softmax(logits)[0, action]
        logging.debug(f"Selected action: {action}, Log probability: {log_prob}")

        return action, log_prob
    except Exception as e:
        logging.error(f"Actor forward pass failed: {str(e)}")
        logging.error(f"State apply_fn: {state.apply_fn}")
        logging.error(f"Actor params: {jax.tree_util.tree_map(lambda x: x.shape if hasattr(x, 'shape') else type(x), actor_params)}")
        logging.error(f"Observation: {observation}")
        raise RuntimeError(f"Actor forward pass failed: {str(e)}") from e

from typing import Dict, Generator, Any

def get_minibatches(data: Dict[str, jnp.ndarray], batch_size: int) -> Generator[Dict[str, jnp.ndarray], None, None]:
    data_size = len(data['obs'])
    assert all(len(v) == data_size for v in data.values()), "All arrays must have the same length"
    indices = np.random.permutation(data_size)
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
) -> Tuple[ExtendedTrainState, List[float], Dict[str, Any]]:
    rng = jax.random.PRNGKey(seed)
    logging.info(f"Initializing PPO agent training with seed {seed}")
    start_time = time.time()

    try:
        dummy_obs, _ = env.reset()
        dummy_obs = jnp.array(dummy_obs)
        tx = optax.adam(learning_rate=learning_rate)
        state = create_train_state(rng, agent, dummy_obs, tx)
        buffer = PPOBuffer(buffer_size, env.observation_space.shape, env.action_space.shape)
        logging.info(f"Initialized agent and PPO buffer with size {buffer_size}")
        logging.info(f"Agent architecture: {agent}")
        logging.info(f"Training parameters: LR={learning_rate}, Gamma={gamma}, Clip ratio={clip_ratio}")
        logging.info(f"Episodes={num_episodes}, Max steps={max_steps}, Batch size={batch_size}")
    except Exception as e:
        logging.error(f"Error initializing PPO agent: {str(e)}")
        raise RuntimeError(f"Failed to initialize PPO agent: {str(e)}")

    @jax.jit
    def update_ppo(state: ExtendedTrainState, batch: Dict[str, jnp.ndarray]):
        def loss_fn(params):
            pi = state.apply_fn({'params': params['actor']}, batch['obs'], method='actor_forward')
            v = state.apply_fn({'params': params['critic']}, batch['obs'], method='critic_forward')

            log_prob = jax.nn.log_softmax(pi)[jnp.arange(batch['act'].shape[0]), batch['act']]
            ratio = jnp.exp(log_prob - batch['logp'])
            clip_adv = jnp.clip(ratio, 1-clip_ratio, 1+clip_ratio) * batch['adv']
            policy_loss = -jnp.mean(jnp.minimum(ratio * batch['adv'], clip_adv))

            value_pred = v.squeeze(-1)
            value_loss = 0.5 * jnp.mean(jnp.square(value_pred - batch['ret']))

            entropy = -jnp.mean(jnp.sum(jax.nn.softmax(pi) * jax.nn.log_softmax(pi), axis=-1))
            kl = jnp.mean(batch['logp'] - log_prob)

            total_loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy

            return total_loss, (policy_loss, value_loss, kl, entropy)

        (total_loss, (policy_loss, value_loss, kl, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, policy_loss, value_loss, kl, entropy

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
                rng, action_rng = jax.random.split(rng)
                action, log_prob = select_action(state, obs, action_rng)
                next_obs, reward, done, truncated, _ = env.step(int(action))

                episode_reward += reward
                episode_length += 1

                value = state.apply_fn({'params': state.params['critic']}, obs[None], method='critic_forward')[0]
                buffer.add(obs, action, reward, value, log_prob)

                if buffer.ptr == buffer.max_size:
                    last_val = state.apply_fn({'params': state.params['critic']}, next_obs[None], method='critic_forward')[0]
                    buffer.finish_path(last_val)
                    data = buffer.get()
                    for _ in range(n_epochs):
                        for mini_batch in get_minibatches(data, batch_size):
                            state, policy_loss, value_loss, kl, entropy = update_ppo(state, mini_batch)
                            training_info['policy_loss_history'].append(float(policy_loss))
                            training_info['value_loss_history'].append(float(value_loss))
                            training_info['kl_history'].append(float(kl))
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
    return state, episode_rewards, training_info

def run_validation(agent, env, num_episodes: int, max_steps: int) -> List[float]:
    validation_rewards = []
    for _ in range(num_episodes):
        timestep = env.reset()
        episode_reward = 0
        for _ in range(max_steps):
            action = agent.select_action(timestep.observation)
            timestep = env.step(action)
            episode_reward += timestep.reward
            if timestep.last():
                break
        validation_rewards.append(episode_reward)
    return validation_rewards

def run_validation(state: ExtendedTrainState, env: RLEnvironment, num_episodes: int, max_steps: int) -> List[float]:
    validation_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        rng = jax.random.PRNGKey(int(time.time()))  # Create a new RNG key for each episode
        for _ in range(max_steps):
            rng, action_rng = jax.random.split(rng)
            action, _ = select_action(state, obs, action_rng)
            obs, reward, done, truncated, _ = env.step(int(action))
            episode_reward += reward
            if done or truncated:
                break
        validation_rewards.append(episode_reward)
    return validation_rewards

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, alpha: float = 0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        max_priority = np.max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float = 0.4):
        if self.size == 0:
            return None

        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(self.size, batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones), indices, weights)

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return self.size
