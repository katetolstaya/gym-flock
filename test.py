import gym
import gym_flock
import configparser
import numpy as np


env_name = "Shepherding-v0"
env = gym.make(env_name)

N = 1000
total_reward = 0
for _ in range(N):
    _ = env.reset()
    episode_reward = 0
    done = False
    while not done:
        action = env.env.controller()
        _, reward, done, _ = env.step(action)
        # state = next_state
        env.render()
env.close()
