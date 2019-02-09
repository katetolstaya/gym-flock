# gym_flock

## Dependencies
- OpenAI [Gym](https://github.com/openai/gym) 0.11.0

## To use
1) clone
2) pip3 install -e . (python2 doesnt work)
3) import gym
4) import gym_flock
5) env = gym.make("Flocking-v0") or env = gym.make("LQR-v0")

Note that the `observation_space` and `action_space` are for a single agent. Let's say there are N agents, K features per agent observation and M actions per agent. Then, `observation_space` is `(K,)` and `action_space` is `(M,)`. 
But `step()` and `reset()` return observations for all agents at once with dimension `(N,K)`. The `step()` function takes an action with dimension `(N,M)`.

