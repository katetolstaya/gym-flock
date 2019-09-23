import numpy as np
from gym_flock.envs.flocking_relative import FlockingRelativeEnv


class FlockingEnv(FlockingRelativeEnv):

    def __init__(self):
        super(FlockingEnv, self).__init__()

    def step(self, u):
        super(FlockingEnv, self).step(u)
        return (self.x, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingEnv, self).reset()
        return (self.x, self.state_network)