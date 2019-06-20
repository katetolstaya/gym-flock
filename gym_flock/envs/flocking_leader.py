import gym
import numpy as np
import matplotlib.pyplot as plt
from gym_flock.envs.flocking_relative import FlockingRelativeEnv

# def grid(N, side=5):
#     side2 = int(N / side)
#     xs = np.arange(0, side) - side / 2.0
#     ys = np.arange(0, side2) - side2 / 2.0
#     xs, ys = np.meshgrid(xs, ys)
#     xs = xs.reshape((N, 1))
#     ys = ys.reshape((N, 1))
#     return 0.8 * np.hstack((xs, ys))


class FlockingLeaderEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingLeaderEnv, self).__init__()
        self.n_leaders = 2
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0
        self.quiver = None
        self.half_leaders  = int(self.n_leaders/2.0)


    def params_from_cfg(self, args):
        super(FlockingLeaderEnv, self).params_from_cfg(args)
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_leaders] = 0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        #u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u

        # x, y position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5 * self.mask
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5 * self.mask
        # x, y velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt * self.mask
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt * self.mask

        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingLeaderEnv, self).reset()
        self.x[0:self.n_leaders,2:4] = np.ones((self.n_leaders, 2)) * np.random.uniform(low=-self.v_max, high=self.v_max, size=(1,1))
        return (self.state_values, self.state_network)

    # def reset(self):
    #     self.x = np.zeros((self.n_agents, self.nx_system))

    #     self.x[:,0:2] = grid(self.n_agents)
    #     self.x[:,2:4] = [0, 1.0]

    #     self.x[0,:] = [1.9, 0.5, 1.0, 0]
    #     self.x[1,:] = [1.9, -0.5, 1.0, 0]

    #     # keep good initialization
    #     self.mean_vel = np.mean(self.x[:, 2:4], axis=0)
    #     self.init_vel = self.x[:, 2:4]
    #     #self.a_net = self.get_connectivity(self.x)
    #     self.compute_helpers()
    #     return (self.state_values, self.state_network)

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

