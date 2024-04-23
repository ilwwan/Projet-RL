import torch
import torch.nn as nn
import torch.optim as optim
from models.DQN import ReplayBuffer
import numpy as np
from copy import deepcopy


class CNNet(nn.Module):
    def __init__(self, observation_shape, hidden_size, num_actions):
        super(CNNet, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(observation_shape[0], 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(self.calculate_conv_output_size(observation_shape), hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

    def calculate_conv_output_size(self, observation_shape):
        conv_input = torch.zeros(1, *observation_shape)
        # Get the output of the convolutional layers
        conv_output = self.conv_layers(conv_input)
        # Calculate the size of the output for a single sample
        conv_output_size = conv_output.view(conv_output.size(0), -1).size(1)
        return conv_output_size


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
        hidden_size,
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
        self.hidden_size = hidden_size

        self.epsilon_start = epsilon_start
        self.decrease_epsilon_factor = (
            decrease_epsilon_factor  # larger -> more exploration
        )
        self.epsilon_min = epsilon_min

        self.learning_rate = learning_rate

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
        self.q_net = CNNet(
            self.observation_space.shape, self.hidden_size, self.action_space.n
        ).to(self.device)
        self.target_net = CNNet(
            self.observation_space.shape, self.hidden_size, self.action_space.n
        ).to(self.device)

    def get_q(self, state):
        """
        Compute Q function for a states
        """
        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
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
