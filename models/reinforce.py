import torch
import torch.nn.functional as F

from .DQN import Net


class ReinforceAgent:
    def __init__(
        self,
        action_space,
        observation_space,
        gamma,
        episode_batch_size,
        learning_rate,
        net_arch=[128],
    ):
        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma

        self.episode_batch_size = episode_batch_size
        self.learning_rate = learning_rate
        self.net_arch = net_arch

        self.reset()

    def get_action(self, state):
        state_tensor = torch.tensor(state).unsqueeze(0)
        with torch.no_grad():
            y = self.policy_net.forward(state_tensor)

            mean = y[:, : self.action_space.shape[0]]
            std = F.sigmoid(y[:, self.action_space.shape[0] :]) + 1e-5

            normal_distributions = torch.distributions.Normal(mean, std)
            sample = normal_distributions.sample()
            return sample.numpy().squeeze()

    def update(self, state, action, reward, terminated, next_state):
        self.current_episode.append(
            (
                torch.tensor(state).unsqueeze(0),
                torch.tensor([[action]]),
                torch.tensor([reward]),
            )
        )
        if terminated:
            self.n_eps += 1

            states, actions, rewards = [
                torch.cat(data) for data in zip(*self.current_episode)
            ]

            current_episode_returns = self._gradiens_returns(rewards, self.gamma)

    def reset(self):
        n_obs = self.observation_space.shape[0]
        n_outputs = self.action_space.shape[0] * 2
        self.policy_net = Net(n_obs, self.net_arch, n_outputs)

        self.scores = []
        self.current_episode = []
        self.optimizer = torch.optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate
        )
        self.n_eps = 0
