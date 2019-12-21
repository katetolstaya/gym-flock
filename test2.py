import gym
env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    print('Remove {} from registry'.format(env))
    del gym.envs.registration.registry.env_specs[env]

import gym_flock
import configparser
import numpy as np

# Initialize the gym environment
env_name = "MappingRad-v0"
# env_name = "Shepherding-v0"
env = gym.make(env_name)
keys = ['nodes', 'edges', 'senders', 'receivers']
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)
env.env.env.local = False

# Run N episodes
N = 10

# for each episode
for _ in range(N):
    # reset the environment
    obs = env.reset()
    episode_reward = 0

    # simulate episode until done
    done = False
    while not done:
        # compute the baseline controller
        action = env.env.env.controller()

        # simulate one step of the environment
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # visualize the environment
        env.render()
    print(episode_reward)
env.close()
