import torch
import numpy as np
import random
from functools import reduce
from copy import deepcopy

# Imports
import torch.nn as nn
import torch.optim as optim

# import os
# os.environ["SDL_VIDEODRIVER"] = "dummy"
# from IPython.display import clear_output


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, terminated, next_state):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, terminated, next_state)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.choices(self.memory, k=batch_size)

    def __len__(self):
        return len(self.memory)


class Net(nn.Module):
    """
    Basic neural net.
    """

    def __init__(self, obs_size, net_arch, n_actions):
        super(Net, self).__init__()
        layers = []
        prev_size = obs_size
        for hidden_size in net_arch:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        self.net = nn.Sequential(*layers[:-1], nn.Linear(prev_size, n_actions))

    def forward(self, x):
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


class DQN:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        batch_size,
        buffer_capacity,
        update_target_every,
        epsilon_start,
        decrease_epsilon_factor,
        epsilon_min,
        learning_rate,
        net_arch,
    ):
        # Check if CUDA is available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.batch_size = batch_size
        self.buffer_capacity = buffer_capacity
        self.update_target_every = update_target_every

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate
        self.net_arch = net_arch

        self.reset()

    def update(self, state, action, reward, terminated, next_state):
        """
        ** SOLUTION **
        """

        # add data to replay buffer
        self.buffer.push(
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device),
            torch.tensor([[action]], dtype=torch.int64).to(self.device),
            torch.tensor([reward]).to(self.device),
            torch.tensor([terminated], dtype=torch.int64).to(self.device),
            torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device),
        )

        if len(self.buffer) < self.batch_size:
            return np.inf

        # get batch
        transitions = self.buffer.sample(self.batch_size)

        (
            state_batch,
            action_batch,
            reward_batch,
            terminated_batch,
            next_state_batch,
        ) = tuple([torch.cat(data).to(self.device) for data in zip(*transitions)])
        next_state_batch = next_state_batch.float()

        values = self.q_net.forward(state_batch).gather(1, action_batch)

        # Compute the ideal Q values
        with torch.no_grad():
            next_state_values = (1 - terminated_batch) * self.target_net(
                next_state_batch
            ).max(1)[0]
            reward_batch = reward_batch.float()
            targets = next_state_values * self.gamma + reward_batch

        loss = self.loss_function(values, targets.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        if not ((self.n_steps + 1) % self.update_target_every):
            self.target_net.load_state_dict(self.q_net.state_dict())

        self.decrease_epsilon()

        self.n_steps += 1
        if terminated:
            self.n_eps += 1

        return loss.detach().cpu().numpy()

    def get_action(self, state, epsilon=None):
        """
        Return action according to an epsilon-greedy exploration policy
        """
        if epsilon is None:
            epsilon = self.epsilon

        if np.random.rand() < epsilon:
            return self.action_space.sample()
        else:
            return np.argmax(self.get_q(state))

    def decrease_epsilon(self):
        self.epsilon = self.epsilon_min + (self.epsilon_start - self.epsilon_min) * (
            np.exp(-1.0 * self.n_eps / self.decrease_epsilon_factor)
        )

    def reset(self):

        self.buffer = ReplayBuffer(self.buffer_capacity)
        self.build_nets()

        self.loss_function = nn.MSELoss()
        self.optimizer = optim.Adam(
            params=self.q_net.parameters(), lr=self.learning_rate
        )

        self.epsilon = self.epsilon_start
        self.n_steps = 0
        self.n_eps = 0

    def build_nets(self):
        if hasattr(self.net_arch, "__iter__"):
            obs_size = product(self.observation_space.shape)
            n_actions = self.action_space.n
            self.q_net = Net(obs_size, self.net_arch, n_actions).to(self.device)
            self.target_net = Net(obs_size, self.net_arch, n_actions).to(self.device)
        else:
            self.q_net = deepcopy(self.net_arch).to(self.device)
            self.target_net = deepcopy(self.net_arch).to(self.device)

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = (
            torch.tensor(state.flatten(), dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            output = self.q_net.forward(state_tensor)  # shape (1,  n_actions)
        return output.cpu().numpy()[0]  # shape  (n_actions)

    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(torch.load(path))
        self.q_net.eval()


def product(iterable):
    return reduce((lambda x, y: x * y), iterable, 1)
