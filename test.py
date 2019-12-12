import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    print('Remove {} from registry'.format(env))
    del gym.envs.registration.registry.env_specs[env]

import gym_flock
import configparser
import numpy as np

# Initialize the gym environment
env_name = "Shepherding-v0"
env = gym.make(env_name)

# Run N episodes
N = 10

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
