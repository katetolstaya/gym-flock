# gym_flock

## Dependencies
- OpenAI [Gym](https://github.com/openai/gym) 0.11.0

## To install
1) Clone this repository
2) `pip3 install -e . (python2 doesnt work)`

## To use

Include the following code in your Python script:
~~~~
import gym  
import gym_flock 
env = gym.make("FlockingRelative-v0")` 
~~~~
and then use the `env.reset()` and `env.step()` as normal. These implementations also include a `env.controller()` function that gives the best current set of actions to be used for imitation learning.




