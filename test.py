import gym
import gym_flock
import argparse
import timeit

parser = argparse.ArgumentParser(description="My parser")
parser.add_argument('-g', '--greedy', dest='greedy', action='store_true')
parser.add_argument('-e', '--expert', dest='expert', action='store_true')
parser.add_argument('-r', '--render', dest='render', action='store_true')
parser.add_argument('-f', '--full', dest='full', action='store_true')
parser.add_argument('-n', '--n', nargs='?', const=20, type=int)

parser.set_defaults(greedy=False, expert=False, render=False, n=20, full=False)

args = parser.parse_args()

# Initialize the gym environment
if args.full:
    env_name = "CoverageFull-v0"
else:
    env_name = "CoverageARL-v0"

env = gym.make(env_name)
env = gym.wrappers.FlattenDictWrapper(env, dict_keys=env.env.keys)

# Run N episodes
N = args.n
total_reward = 0

start_time = timeit.default_timer()

# for each episode
for _ in range(N):
    # reset the environment
    obs = env.reset()
    episode_reward = 0

    # simulate episode until done
    done = False
    while not done:
        # compute the baseline controller
        if args.expert:
            try:
                action = env.env.env.controller(random=False, greedy=False, reset_solution=False)
            except AssertionError:
                obs = env.reset()
                episode_reward = 0
                done = False
                continue
        elif args.greedy:
            action = env.env.env.controller(random=False, greedy=True)
        else:
            action = env.env.env.controller(random=True)

        # simulate one step of the environment
        obs, reward, done, _ = env.step(action)
        episode_reward += reward

        if args.render:  # visualize the environment
            env.render()

    print(episode_reward)
    total_reward += episode_reward

elapsed = timeit.default_timer() - start_time

print('Average reward: ' + str(total_reward / N))
print('Elapsed time: ' + str(elapsed))

env.close()
