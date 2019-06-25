import numpy as np
from gym_flock.envs.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.utils import twoflocks, grid


class FlockingTwoFlocksEnv(FlockingRelativeEnv):

    def reset(self):
        self.x = np.zeros((self.n_agents, self.nx_system))
        # grids, vels = twoflocks(self.n_agents, delta=self.n_agents/10*0.8+0.25, side=5)
        # self.x[:, 0:2] = grids
        # self.x[:, 2:4] = vels * 0.25
        # self.x[:, 2] = np.random.uniform(low=-self.v_max*0.25, high=self.v_max*0.25, size=(self.n_agents,))

        bias = np.random.uniform(low=-self.v_bias/2.0, high=self.v_bias/2.0, size=(2,))
        scale = 0.1
        grids = grid(self.n_agents, side=int(self.n_agents/10))
        self.x[:, 0:2] = grids
        self.x[:, 2:4] = -grids
        self.x[:, 2] = self.x[:, 2] + bias[0]
        self.x[:, 3] = self.x[:, 3]  + bias[1]

        self.mean_vel = np.mean(self.x[:, 2:4], axis=0)
        self.init_vel = self.x[:, 2:4]
        self.compute_helpers()
        return (self.state_values, self.state_network)
