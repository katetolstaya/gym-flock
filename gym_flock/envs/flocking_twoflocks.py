import numpy as np
from gym_flock.envs.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.utils import twoflocks


class FlockingTwoFlocksEnv(FlockingRelativeEnv):

    def reset(self):
        self.x = np.zeros((self.n_agents, self.nx_system))
        grids, vels = twoflocks(self.n_agents, delta=4, side=int(self.n_agents/10))
        self.x[:, 0:2] = grids
        self.x[:, 2:4] = vels * 0.5
        self.x[:, 2] = np.random.uniform(low=-self.v_max*0.25, high=self.v_max*0.25, size=(self.n_agents,))
        self.mean_vel = np.mean(self.x[:, 2:4], axis=0)
        self.init_vel = self.x[:, 2:4]
        self.compute_helpers()
        return (self.state_values, self.state_network)
