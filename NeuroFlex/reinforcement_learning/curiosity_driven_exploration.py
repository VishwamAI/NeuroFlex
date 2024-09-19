"""
Curiosity-Driven Exploration Module

This module implements curiosity-driven exploration methods for reinforcement learning,
including the Intrinsic Curiosity Module (ICM) and novelty detection mechanisms.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class ICM(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64, beta=0.2):
        super().__init__()
        self.beta = beta

        self.feature_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.inverse_model = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        self.forward_model = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state, next_state, action):
        state_feat = self.feature_encoder(state)
        next_state_feat = self.feature_encoder(next_state)

        pred_action = self.inverse_model(torch.cat([state_feat, next_state_feat], dim=1))
        pred_next_state_feat = self.forward_model(torch.cat([state_feat, action], dim=1))

        return pred_action, pred_next_state_feat, next_state_feat

    def compute_intrinsic_reward(self, state, next_state, action):
        with torch.no_grad():
            _, pred_next_state_feat, next_state_feat = self(state, next_state, action)
            intrinsic_reward = self.beta * 0.5 * torch.mean((pred_next_state_feat - next_state_feat)**2, dim=1)
        return intrinsic_reward.item()

    def update(self, state, next_state, action):
        pred_action, pred_next_state_feat, next_state_feat = self(state, next_state, action)

        inverse_loss = nn.MSELoss()(pred_action, action)
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
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

    def compute_total_reward(self, state, next_state, action, extrinsic_reward):
        intrinsic_reward = self.icm.compute_intrinsic_reward(
            torch.FloatTensor(state).unsqueeze(0),
            torch.FloatTensor(next_state).unsqueeze(0),
            torch.FloatTensor(action).unsqueeze(0)
        )

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
        episode_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, extrinsic_reward, done, _ = env.step(action)

            total_reward = agent.compute_total_reward(state, next_state, action, extrinsic_reward)
            episode_reward += total_reward

            agent.update(state, action, total_reward, next_state, done)

            state = next_state

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    return agent

def main():
    import gym
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = CuriosityDrivenAgent(state_dim, action_dim)
    trained_agent = train_curiosity_driven_agent(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = trained_agent.act(state)
        next_state, reward, done, _ = env.step(np.argmax(action))
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
