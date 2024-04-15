from models.DQN import DQN
import gymnasium as gym
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()


def train(env, agent, N_episodes, eval_every=10, reward_threshold=200):
    total_time = 0
    state, _ = env.reset()
    losses = []
    for episode in range(N_episodes):
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            loss_val = agent.update(state, action, reward, terminated, next_state)

            state = next_state
            losses.append(loss_val)

            done = terminated or truncated
            total_time += 1
        if (episode + 1) % eval_every == 0:
            rewards = eval_agent(agent, env)
            print(f"Episode {episode+1} - Reward: {np.mean(rewards)}")
            if np.mean(rewards) >= reward_threshold:
                print(f"Task solved in {episode} episodes!")
                break

        # Log the loss value
        writer.add_scalar("Loss", loss_val, episode)

        # Log other metrics as needed

    # Close the SummaryWriter
    writer.close()

    return losses


def eval_agent(agent, env, n_sim=5):
    """
    ** Solution **

    Monte Carlo evaluation of DQN agent.

    Repeat n_sim times:
        * Run the DQN policy until the environment reaches a terminal state (= one episode)
        * Compute the sum of rewards in this episode
        * Store the sum of rewards in the episode_rewards array.
    """
    env_copy = deepcopy(env)
    episode_rewards = np.zeros(n_sim)
    for i in range(n_sim):
        state, _ = env_copy.reset()
        reward_sum = 0
        done = False
        while not done:
            action = agent.get_action(state, 0)
            state, reward, terminated, truncated, _ = env_copy.step(action)
            reward_sum += reward
            done = terminated or truncated
        episode_rewards[i] = reward_sum
    return episode_rewards


env = gym.make("CartPole-v1", render_mode="rgb_array")

action_space = env.action_space
observation_space = env.observation_space

gamma = 0.99
batch_size = 128
buffer_capacity = 10_000
update_target_every = 32

epsilon_start = 0.9
decrease_epsilon_factor = 1000
epsilon_min = 0.05

learning_rate = 1e-1

arguments = (
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
    [32, 32],
)

N_episodes = 300

agent = DQN(*arguments)

# Run the training loop
losses = train(env, agent, N_episodes, reward_threshold=500)

plt.plot(losses)

# Evaluate the final policy
rewards = eval_agent(agent, env, 20)
print("")
print("mean reward after training = ", np.mean(rewards))
plt.show()
