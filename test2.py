import gym
import gym_flock
import configparser
import numpy as np
import time

# Initialize the gym environment
env_name = "MappingRad-v0"
# env_name = "Shepherding-v0"
env = gym.make(env_name)
keys = ['nodes', 'edges', 'senders', 'receivers']
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=keys)

# Run N episodes
N = 100
total_reward = 0

optimal = True


# for each episode
for _ in range(N):
    # reset the environment
    obs = env.reset()
    episode_reward = 0

    # simulate episode until done
    done = False
    while not done:
        # compute the baseline controller
        if optimal:
            try:
                action = env.env.env.controller()
            except AssertionError:
                obs = env.reset()
                episode_reward = 0
                done = False
                continue
        else:
            # action = env.env.env.controller(random=True)
            action = env.env.env.controller(random=False, greedy=True)

        # simulate one step of the environment
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        # visualize the environment
        env.render()
        time.sleep(0.125)

    print(episode_reward)
    total_reward += episode_reward

print(total_reward / N)

env.close()
