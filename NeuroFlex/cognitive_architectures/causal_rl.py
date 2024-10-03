# causal_rl.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

class CausalRLAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(CausalRLAgent, self).__init__()
        self.fc = nn.Linear(state_size, action_size)

    def forward(self, x):
        return self.fc(x)

    def select_action(self, state):
        logits = self.forward(state)
        return Categorical(logits=logits).sample()

def train_causal_rl(agent, environment, episodes, gamma=0.99):
    optimizer = optim.Adam(agent.parameters())
    for episode in range(episodes):
        state = environment.reset()
        done = False
        log_probs = []
        rewards = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = environment.step(action)

            # Calculate log probability of the action
            log_prob = Categorical(logits=agent.forward(state)).log_prob(action)

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Calculate discounted rewards
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)

        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Calculate loss
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()

        # Backpropagate and optimize
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

    return agent
