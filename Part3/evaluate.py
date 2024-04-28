import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
import config_1 as config
import numpy as np

env = config.env
# Load the trained PPO model
model = PPO.load("highway_ppo")

# Wrap the environment in a vectorized environment
env = DummyVecEnv([lambda: env])

# Evaluate the model in the environment
ppo_mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1000)

# Define a random policy function


def random_policy(obs):
    return np.random.randint(env.action_space.n)


# Evaluate the random policy for 1000 episodes
random_rewards = []
for _ in range(1000):
    obs = env.reset()
    total_reward = 0
    done = False
    while not done:
        action = random_policy(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
    random_rewards.append(total_reward)

# Calculate mean reward of random policy
random_mean_reward = np.mean(random_rewards)

# Print the mean rewards of both policies
print(f"Mean reward of PPO model: {ppo_mean_reward}")
print(f"Mean reward of random policy: {random_mean_reward}")

# Close the environment
env.close()
