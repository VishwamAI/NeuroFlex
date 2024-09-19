"""
Curiosity-Driven Exploration Module

This module implements curiosity-driven exploration methods for reinforcement learning,
including the Intrinsic Curiosity Module (ICM) and novelty detection mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, beta=0.2):
        super().__init__()
        self.beta = beta
        self.action_dim = action_dim  # Use the provided action_dim

        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + self.action_dim, hidden_dim),  # Use self.action_dim
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        # Ensure action is the correct shape before one-hot encoding
        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)
        if action.dim() == 0:
            action = action.unsqueeze(0)
        elif action.dim() > 1:
            action = action.squeeze()

        # Convert action to long tensor and clamp values
        action = action.long()
        action = torch.clamp(action, 0, self.action_dim - 1)

        # Apply one-hot encoding
        action_one_hot = F.one_hot(action, num_classes=self.action_dim).float()

        # Reshape action tensor to match state_feat dimensions
        action_one_hot = action_one_hot.view(state_feat.size(0), -1)

        # Verify dimensions and ensure action_one_hot is correct
        print(f"state_feat shape: {state_feat.shape}, action_one_hot shape: {action_one_hot.shape}")
        if action_one_hot.size(1) != self.action_dim:
            raise ValueError(f"Action one-hot encoding size mismatch. Expected {self.action_dim}, got {action_one_hot.size(1)}")

        # Ensure the concatenated input matches the expected size
        expected_input_size = self.forward_model[0].in_features
        current_input_size = state_feat.size(1) + action_one_hot.size(1)

        if current_input_size != expected_input_size:
            raise ValueError(f"Input size mismatch. Expected {expected_input_size}, got {current_input_size}")

        print(f"Final input size: {state_feat.size(1) + action_one_hot.size(1)}")

        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action_one_hot], dim=1))
        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))

        # Ensure pred_action matches the size of action_one_hot
        if pred_action.size(1) != self.action_dim:
            raise ValueError(f"Predicted action size mismatch. Expected {self.action_dim}, got {pred_action.size(1)}")

        return pred_action, pred_next_state_feat, next_state_feat

    def compute_intrinsic_reward(self, state, next_state, action):
        with torch.no_grad():
            _, pred_next_state_feat, next_state_feat = self(state, next_state, action)
            intrinsic_reward = self.beta * 0.5 * torch.mean((pred_next_state_feat - next_state_feat)**2, dim=1)
        return intrinsic_reward.item() if intrinsic_reward.numel() == 1 else intrinsic_reward.mean().item()

    def update(self, state, next_state, action):
        pred_action, pred_next_state_feat, next_state_feat = self(state, next_state, action)

        # Ensure pred_action and action have the same size
        if pred_action.dim() > 0 and action.dim() > 0:
            if pred_action.size(0) != action.size(0):
                pred_action = pred_action[:action.size(0)]  # Truncate if necessary
        else:
            # Handle the case where either tensor has no dimensions
            pred_action = pred_action.view(-1)
            action = action.view(-1)

        inverse_loss = nn.MSELoss()(pred_action, action.float())
        forward_loss = 0.5 * torch.mean((pred_next_state_feat - next_state_feat.detach())**2, dim=1)

        loss = inverse_loss + forward_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

class NoveltyDetector:
    def __init__(self, state_dim, memory_size=1000, novelty_threshold=0.1):
        self.memory = np.zeros((memory_size, state_dim))
        self.memory_index = 0
        self.memory_size = memory_size
        self.novelty_threshold = novelty_threshold

    def compute_novelty(self, state):
        if self.memory_index < self.memory_size:
            distances = np.mean(np.abs(self.memory[:self.memory_index] - state), axis=1)
        else:
            distances = np.mean(np.abs(self.memory - state), axis=1)

        novelty = np.min(distances)
        return novelty

    def update_memory(self, state):
        self.memory[self.memory_index % self.memory_size] = state
        self.memory_index += 1

    def is_novel(self, state):
        novelty = self.compute_novelty(state)
        return novelty > self.novelty_threshold

class CuriosityDrivenAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )

        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.icm = ICM(state_dim, action_dim)
        self.novelty_detector = NoveltyDetector(state_dim)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def act(self, state):
        state_np = np.array(state[0] if isinstance(state, tuple) else state)
        # print(f"Input state type: {type(state)}, shape: {state_np.shape}")
        state = torch.FloatTensor(state_np).unsqueeze(0)  # Convert to numpy array, then to tensor, and add batch dimension
        # print(f"Processed state type: {type(state)}, shape: {state.shape}")
        action_probs = self.actor(state)
        action = torch.argmax(action_probs, dim=1)
        action_one_hot = F.one_hot(action, num_classes=self.icm.action_dim).float()  # Use self.icm.action_dim for correct size
        return action.item(), action_one_hot.squeeze(0).detach().numpy()  # Return both scalar action and numpy array of one-hot encoded action

    def compute_total_reward(self, state, next_state, action, extrinsic_reward):
        intrinsic_reward = self.icm.compute_intrinsic_reward(
            torch.FloatTensor(state).unsqueeze(0),
            torch.FloatTensor(next_state).unsqueeze(0),
            torch.FloatTensor(action).unsqueeze(0)
        )

        self.novelty_detector.update_memory(next_state)
        novelty = self.novelty_detector.compute_novelty(next_state)
        novelty_reward = novelty if self.novelty_detector.is_novel(next_state) else 0

        total_reward = extrinsic_reward + intrinsic_reward + novelty_reward
        return total_reward

    def update(self, state, action, reward, next_state, done):
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor([reward])
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor([done])

        # Update ICM
        icm_loss = self.icm.update(state.unsqueeze(0), next_state.unsqueeze(0), action.unsqueeze(0))

        # Update novelty detector
        self.novelty_detector.update_memory(next_state.numpy())

        # Compute TD error
        value = self.critic(state)
        next_value = self.critic(next_state)
        td_error = reward + (1 - done) * 0.99 * next_value - value

        # Update critic
        critic_loss = td_error.pow(2).mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(state)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return icm_loss, critic_loss.item(), actor_loss.item()

def train_curiosity_driven_agent(env, agent, num_episodes=1000):
    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the state from the tuple if necessary
        episode_reward = 0
        done = False

        while not done:
            action, action_one_hot = agent.act(state)
            next_state, extrinsic_reward, done, _, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the state from the tuple if necessary

            total_reward = agent.compute_total_reward(state, next_state, action_one_hot, extrinsic_reward)
            episode_reward += total_reward

            agent.update(state, action_one_hot, total_reward, next_state, done)

            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    return agent

def main():
    import gymnasium as gym
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    print(f"Environment action dimension: {action_dim}")

    agent = CuriosityDrivenAgent(state_dim, action_dim)
    print(f"Agent action dimension: {agent.action_dim}")
    print(f"ICM action dimension: {agent.icm.action_dim}")

    trained_agent = train_curiosity_driven_agent(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action, _ = trained_agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
