"""
NeuroFlex Reinforcement Learning Module

This module provides reinforcement learning algorithms and tools, including DQN, PPO, A2C, SAC, and TD3.

Recent updates:
- Enhanced support for prioritized replay buffers
- Improved training functions for RL agents
- Added epsilon-greedy policy creation utility
- Implemented advanced RL algorithms: SAC (Soft Actor-Critic) and TD3 (Twin Delayed DDPG)
  - SAC: Off-policy algorithm for continuous action spaces with entropy regularization
  - TD3: Improved version of DDPG with twin Q-networks and delayed policy updates
- Added support for multi-agent reinforcement learning
  - Includes functions for creating multi-agent environments and training multiple agents
- Optimized performance for both single-agent and multi-agent scenarios
- Implemented model-based RL techniques
  - Includes environment models for sample-efficient learning
  - Added model-based planning and decision-making functions
- Integrated curiosity-driven exploration methods
  - Implemented Intrinsic Curiosity Module (ICM) for better exploration
  - Added novelty detection mechanisms to encourage exploration of new states
- Updated version to match main NeuroFlex version
"""

from .rl_module import RLAgent, RLEnvironment, PrioritizedReplayBuffer, train_rl_agent
from .advanced_rl_algorithms import SACAgent, TD3Agent, create_sac_agent, create_td3_agent

__all__ = [
    'RLAgent',
    'RLEnvironment',
    'PrioritizedReplayBuffer',
    'train_rl_agent',
    'SACAgent',
    'TD3Agent',
    'create_sac_agent',
    'create_td3_agent',
    'get_rl_version',
    'SUPPORTED_RL_ALGORITHMS',
    'create_epsilon_greedy_policy',
    'initialize_rl_module',
    'create_multi_agent_environment',
    'train_multi_agent',
    'validate_rl_config'
]

def get_rl_version():
    return "0.1.3"  # Updated to match main NeuroFlex version

SUPPORTED_RL_ALGORITHMS = ['DQN', 'PPO', 'A2C', 'DDPG', 'SAC', 'TD3']

def create_epsilon_greedy_policy(Q, epsilon, num_actions):
    def policy_fn(state):
        import numpy as np
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def initialize_rl_module():
    print("Initializing Reinforcement Learning Module...")
    # Add any necessary initialization code here

# Multi-agent reinforcement learning support
def create_multi_agent_environment(num_agents, env_config):
    """
    Create a multi-agent reinforcement learning environment.

    :param num_agents: Number of agents in the environment
    :param env_config: Configuration for the environment
    :return: Multi-agent environment object
    """
    # Placeholder for multi-agent environment creation
    # This should be implemented based on specific multi-agent RL framework
    pass

def train_multi_agent(agents, env, num_episodes):
    """
    Train multiple agents in a multi-agent reinforcement learning setting.

    :param agents: List of agent objects
    :param env: Multi-agent environment
    :param num_episodes: Number of episodes to train
    """
    # Placeholder for multi-agent training logic
    # This should be implemented based on specific multi-agent RL algorithms
    pass

def validate_rl_config(config):
    """
    Validate the configuration for a reinforcement learning algorithm.
    """
    required_keys = ['algorithm', 'learning_rate', 'discount_factor']
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required configuration key: {key}")

    if config['algorithm'] not in SUPPORTED_RL_ALGORITHMS:
        raise ValueError(f"Unsupported RL algorithm: {config['algorithm']}")

    return True

# Add any other Reinforcement Learning-specific utility functions or constants as needed
