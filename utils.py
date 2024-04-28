from copy import deepcopy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from functools import reduce


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


def train(env, agent, N_episodes, eval_every=10, reward_threshold=200):
    writer = SummaryWriter()
    total_time = 0
    state, _ = env.reset()
    losses = []
    for episode in range(N_episodes):
        done = False
        state, _ = env.reset()
        while not done:
            action = agent.get_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            loss_val = agent.update(state, action, reward, done, next_state)

            state = next_state
            losses.append(loss_val)

            total_time += 1
        if (episode + 1) % eval_every == 0:
            rewards = eval_agent(agent, env)
            print(f"Episode {episode+1} - Reward: {np.mean(rewards)}")
            # Log other metrics as needed
            writer.add_scalar("ep_reward", np.mean(rewards), episode)
            if np.mean(rewards) >= reward_threshold:
                print(f"Task solved in {episode} episodes!")
                break

        # Log the loss value
        writer.add_scalar("Loss", loss_val, episode)

    # Close the SummaryWriter
    writer.close()
    return losses


def product(iterable):
    return reduce((lambda x, y: x * y), iterable, 1)
