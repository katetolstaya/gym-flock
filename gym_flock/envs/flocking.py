import numpy as np
from gym_flock.envs.flocking_relative import FlockingRelativeEnv


class FlockingEnv(FlockingRelativeEnv):

    def __init__(self):
        super(FlockingEnv, self).__init__()
        self.n_neighbors = 7
        self.n_f = self.nx_system * self.n_neighbors

    def step(self, u):
        super(FlockingEnv, self).step(u)
        return (self.get_observation(), self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingEnv, self).reset()
        return self.get_observation(), self.state_network

    def get_observation(self):
        nearest = np.argsort(self.r2, axis=1)
        observation = np.zeros((self.n_agents, self.n_f))
        for i in range(self.n_neighbors):
            observation[:, i*self.nx_system:(i+1)*self.nx_system] = self.x - self.x[nearest[:, i], :]
        return observation