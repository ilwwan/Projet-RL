import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

import numpy as np


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.mean = nn.Linear(64, output_dim)
        self.log_std = nn.Parameter(
            torch.zeros(output_dim)
        )  # Learnable log standard deviation

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)  # Transform log std to std
        return mean, std


class REINFORCE:
    def __init__(
        self, input_dim, output_dim, learning_rate=1e-4, gamma=0.99, grad_clip=0.5
    ):
        self.policy_network = PolicyNetwork(input_dim, output_dim)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.grad_clip = grad_clip

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, std = self.policy_network(state)
        normal = Normal(mean, std)
        action = normal.sample()
        log_prob = normal.log_prob(action)
        return action.item(), log_prob

    def update(self, rewards, log_probs):
        if len(rewards) < 2:
            return
        discounted_rewards = []
        R = 0
        for r in rewards[::-1]:
            R = r + self.gamma * R
            discounted_rewards.insert(0, R)
        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
            discounted_rewards.std() + 1e-9
        )  # Normalize rewards

        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip)
        self.optimizer.step()


if __name__ == "__main__":
    from config.config_2 import env
    from utils import product

    num_episodes = 100

    agent = REINFORCE(
        input_dim=product(env.observation_space.shape),
        output_dim=env.action_space.shape[0],
    )

    for episode in range(num_episodes):
        state, _ = env.reset()
        rewards = []
        log_probs = []
        done = False
        while not done:
            action, log_prob = agent.select_action(state.flatten())
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
        print(f"Episode {episode} - Reward: {sum(rewards)}")
        agent.update(rewards, log_probs)
    while True:
        state, _ = env.reset()
        rewards = []
        log_probs = []
        done = False
        while not done:
            action, log_prob = agent.select_action(state.flatten())
            next_state, reward, terminated, truncated, _ = env.step([action])
            done = terminated or truncated
            rewards.append(reward)
            log_probs.append(log_prob)
            state = next_state
        print(f"Episode - Reward: {sum(rewards)}")
