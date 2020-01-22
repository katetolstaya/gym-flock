import numpy as np
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv


class FlockingStochasticEnv(FlockingRelativeEnv):

    def __init__(self):
        super(FlockingStochasticEnv, self).__init__()
        self.dt_mean = 0.12
        self.dt_sigma = 0.018
        self.max_accel = 0.5
        self.scale = 6.0

    def step(self, u):
        assert u.shape == (self.n_agents, self.nu)
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u * self.scale
        self.x = self.x * self.scale

        self.dt = np.random.normal(self.dt_mean, self.dt_sigma)

        # x position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5
        # y position
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt

        self.x = self.x / self.scale

        self.compute_helpers()

        return (self.state_values, self.state_network), self.instant_cost(), False, {}


    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """
        controls = super(FlockingStochasticEnv, self).controller(centralized)
        controls = np.clip(controls, -1.0 * self.max_accel, self.max_accel)
        return controls
