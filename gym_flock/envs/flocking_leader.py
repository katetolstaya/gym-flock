import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_flock.envs.flocking_relative import FlockingRelativeEnv

class FlockingLeaderEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv, self).__init__()
        self.n_leaders = 2
        
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None

    def params_from_cfg(self, args):
        super(FlockingLeaderEnv, self).params_from_cfg(args)
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        #u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u

        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt #* self.mask
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt #* self.mask
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.gain * self.u[:, 0] * self.dt * self.mask 
        self.x[:, 3] = self.x[:, 3] + self.gain * self.u[:, 1] * self.dt * self.mask

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingLeaderEnv, self).reset()
        self.x[0:self.n_leaders,2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max, high=self.v_max, size=(1,1))
        return (self.state_values, self.state_network)

    def render(self, mode='human'):
        super(FlockingLeaderEnv, self).render(mode)

        X = self.x[0:self.n_leaders,0]
        Y = self.x[0:self.n_leaders,1]
        U = self.x[0:self.n_leaders,2]
        V = self.x[0:self.n_leaders,3]

        if self.quiver == None:
            self.quiver = self.ax.quiver(X,Y,U,V, color='r')
        else:
            self.quiver.set_offsets(self.x[0:self.n_leaders,0:2])
            self.quiver.set_UVC(U,V)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

