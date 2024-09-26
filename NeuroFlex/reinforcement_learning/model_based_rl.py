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

"""
Model-Based Reinforcement Learning Module

This module implements model-based reinforcement learning algorithms and utilities.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class EnvironmentModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state, action):
        return self.model(torch.cat([state, action], dim=1))

    def train(self, states, actions, next_states):
        predicted_next_states = self(states, actions)
        loss = nn.MSELoss()(predicted_next_states, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

class MBPOAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.env_model = EnvironmentModel(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

    def act(self, state):
        state = torch.FloatTensor(state)
        action = self.actor(state)
        return action.detach().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        Q_values = self.critic(torch.cat([states, actions], dim=1))
        next_actions = self.actor(next_states)
        next_Q_values = self.critic(torch.cat([next_states, next_actions], dim=1))
        target_Q_values = rewards + (1 - dones) * 0.99 * next_Q_values

        critic_loss = nn.MSELoss()(Q_values, target_Q_values.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(torch.cat([states, self.actor(states)], dim=1)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update environment model
        model_loss = self.env_model.train(states, actions, next_states)

        return critic_loss.item(), actor_loss.item(), model_loss

def model_based_planning(agent, initial_state, planning_horizon=5, num_simulations=10):
    best_action_sequence = None
    best_reward = float('-inf')

    for _ in range(num_simulations):
        state = torch.FloatTensor(initial_state)
        action_sequence = []
        total_reward = 0

        for _ in range(planning_horizon):
            action = agent.act(state)
            action_sequence.append(action)

            next_state = agent.env_model(state.unsqueeze(0), torch.FloatTensor(action).unsqueeze(0)).squeeze(0)
            reward = agent.critic(torch.cat([state, torch.FloatTensor(action)], dim=0)).item()

            total_reward += reward
            state = next_state

        if total_reward > best_reward:
            best_reward = total_reward
            best_action_sequence = action_sequence

    return best_action_sequence[0]  # Return the first action of the best sequence

def train_mbpo(env, agent, num_episodes=1000, planning_horizon=5, num_simulations=10):
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = model_based_planning(agent, state, planning_horizon, num_simulations)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            agent.update([state], [action], [reward], [next_state], [done])

            state = next_state

            if done:
                break

        if episode % 10 == 0:
            print(f"Episode {episode}, Reward: {episode_reward:.2f}")

    return agent

def main():
    # Example usage
    import gym
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = MBPOAgent(state_dim, action_dim)
    trained_agent = train_mbpo(env, agent)

    # Test the trained agent
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = trained_agent.act(state)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state

    print(f"Test episode reward: {total_reward:.2f}")

if __name__ == "__main__":
    main()
