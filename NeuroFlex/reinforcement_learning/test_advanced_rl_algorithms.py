"""
Test script for advanced reinforcement learning algorithms (SAC and TD3)
"""

import jax
import jax.numpy as jnp
import numpy as np
from advanced_rl_algorithms import SACAgent, TD3Agent, create_sac_agent, create_td3_agent

def test_sac_agent():
    print("Testing SAC Agent...")
    state_dim = 4
    action_dim = 2

    # Initialize SAC agent
    sac_agent = create_sac_agent(state_dim, action_dim)

    # Test action selection
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = sac_agent.select_action(state)
    assert action.shape == (action_dim,), f"Expected action shape {(action_dim,)}, got {action.shape}"

    # Test agent update
    batch_size = 32
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size, 1))

    critic_loss, actor_loss = sac_agent.update((states, actions, rewards, next_states, dones))
    assert isinstance(critic_loss, float), f"Expected float for critic_loss, got {type(critic_loss)}"
    assert isinstance(actor_loss, float), f"Expected float for actor_loss, got {type(actor_loss)}"

    print("SAC Agent tests passed!")

def test_td3_agent():
    print("Testing TD3 Agent...")
    state_dim = 4
    action_dim = 2

    # Initialize TD3 agent
    td3_agent = create_td3_agent(state_dim, action_dim)

    # Test action selection
    state = jnp.array([0.1, -0.2, 0.3, -0.4])
    action = td3_agent.select_action(state)
    assert action.shape == (action_dim,), f"Expected action shape {(action_dim,)}, got {action.shape}"

    # Test agent update
    batch_size = 32
    states = jax.random.normal(jax.random.PRNGKey(0), (batch_size, state_dim))
    actions = jax.random.normal(jax.random.PRNGKey(1), (batch_size, action_dim))
    rewards = jax.random.normal(jax.random.PRNGKey(2), (batch_size, 1))
    next_states = jax.random.normal(jax.random.PRNGKey(3), (batch_size, state_dim))
    dones = jax.random.bernoulli(jax.random.PRNGKey(4), 0.1, (batch_size, 1))

    critic_loss, actor_loss = td3_agent.update((states, actions, rewards, next_states, dones), step=0)
    assert isinstance(critic_loss, float), f"Expected float for critic_loss, got {type(critic_loss)}"
    assert isinstance(actor_loss, float), f"Expected float for actor_loss, got {type(actor_loss)}"

    print("TD3 Agent tests passed!")

if __name__ == "__main__":
    test_sac_agent()
    test_td3_agent()
    print("All tests passed successfully!")
