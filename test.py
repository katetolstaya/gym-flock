import gym
import gym_flock
import configparser
import numpy as np

# Initialize the gym environment
env_name = "Shepherding-v0"
env = gym.make(env_name)

# Run N episodes
N = 1000

# for each episode
for _ in range(N):
    # reset the environment
    observation, graph = env.reset()
    episode_reward = 0

    # simulate episode until done
    done = False
    while not done:
        # compute the baseline controller
        action = env.env.controller()

        # simulate one step of the environment
        (observation, graph), reward, done, _ = env.step(action)

        # visualize the environment
        env.render()
env.close()
