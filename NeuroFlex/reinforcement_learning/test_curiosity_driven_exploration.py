import gymnasium as gym
import numpy as np
import torch
from curiosity_driven_exploration import (
    CuriosityDrivenAgent,
    train_curiosity_driven_agent,
)


def test_curiosity_driven_exploration():
    # Create a simple environment
    env = gym.make("MountainCar-v0")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    print(f"Environment action dimension: {action_dim}")

    # Initialize the CuriosityDrivenAgent
    agent = CuriosityDrivenAgent(state_dim, action_dim)

    # Train the agent
    num_episodes = 100
    trained_agent = train_curiosity_driven_agent(env, agent, num_episodes)

    # Test the trained agent
    test_episodes = 10
    total_rewards = []

    for _ in range(test_episodes):
        state = env.reset()
        print(f"Initial state type: {type(state)}, shape: {np.shape(state)}")
        if isinstance(state, tuple):
            state = state[0]  # Extract the state from the tuple if necessary
        state = np.array(state).flatten()  # Ensure state is a flat numpy array
        print(f"Processed state type: {type(state)}, shape: {state.shape}")
        episode_reward = 0
        done = False

        while not done:
            action = trained_agent.act(state)
            next_state, reward, done, _ = env.step(np.argmax(action))
            print(f"Next state type: {type(next_state)}, shape: {np.shape(next_state)}")
            if isinstance(next_state, tuple):
                next_state = next_state[
                    0
                ]  # Extract the state from the tuple if necessary
            episode_reward += reward
            state = np.array(
                next_state
            ).flatten()  # Ensure next_state is a flat numpy array
            print(f"Processed next state type: {type(state)}, shape: {state.shape}")

        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print(f"Average test episode reward: {avg_reward:.2f}")

    # Test ICM
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]  # Extract the state from the tuple if necessary
    state = np.array(state).flatten()  # Ensure state is a flat numpy array
    next_state, _, _, _ = env.step(env.action_space.sample())
    if isinstance(next_state, tuple):
        next_state = next_state[0]  # Extract the state from the tuple if necessary
    next_state = np.array(
        next_state
    ).flatten()  # Ensure next_state is a flat numpy array
    action = trained_agent.act(state)

    intrinsic_reward = trained_agent.icm.compute_intrinsic_reward(
        torch.FloatTensor(state).unsqueeze(0),
        torch.FloatTensor(next_state).unsqueeze(0),
        torch.FloatTensor(action).unsqueeze(0),
    )
    print(f"Intrinsic reward: {intrinsic_reward:.4f}")

    # Test novelty detection
    novelty = trained_agent.novelty_detector.compute_novelty(next_state)
    is_novel = trained_agent.novelty_detector.is_novel(next_state)
    print(f"Novelty: {novelty:.4f}")
    print(f"Is novel: {is_novel}")

    # Verify that the agent improves over time
    initial_rewards = total_rewards[:5]
    final_rewards = total_rewards[-5:]
    print(f"Initial rewards: {initial_rewards}")
    print(f"Final rewards: {final_rewards}")

    if np.mean(final_rewards) > np.mean(initial_rewards):
        print("Agent shows improvement over time.")
    else:
        print(
            "Agent does not show significant improvement. Further tuning may be required."
        )


if __name__ == "__main__":
    test_curiosity_driven_exploration()
