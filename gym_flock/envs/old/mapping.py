import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class MappingEnv(gym.Env):

    def __init__(self):

        # config_file = path.join(path.dirname(__file__), "params_flock.cfg")
        # config = configparser.ConfigParser()
        # config.read(config_file)
        # config = config['flock']

        self.nearest_agents = 7
        self.nearest_targets = 7

        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not
        self.centralized = True

        # number states per agent
        self.nx_system = 4
        # number of actions per agent
        self.nu = 2

        # default problem parameters
        self.n_agents = 100  # int(config['network_size'])
        # self.comm_radius = 0.9  # float(config['comm_radius'])
        self.dt = 0.1  # #float(config['system_dt'])
        self.v_max = 5.0  # float(config['max_vel_init'])

        self.v_bias = self.v_max

        # intitialize state matrices
        self.x = None
        self.u = None
        self.mean_vel = None
        self.init_vel = None
        self.greedy_action = None

        self.diff = None
        self.r2 = None
        self.adj_mat = None
        self.adj_mat_mean = None

        self.diff_targets = None
        self.r2_targets = None

        self.target_observed = None
        self.state_network = None
        self.state_values = None
        self.reward = None

        self.max_accel = 1

        # self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
        #                                dtype=np.float32)
        #
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, ),
        #                                     dtype=np.float32)

        # target initialization
        self.px_max = 100
        self.py_max = 100
        x = np.linspace(-1.0 * self.px_max, self.px_max, self.n_agents)
        y = np.linspace(-1.0 * self.py_max, self.py_max, self.n_agents)

        tx, ty = np.meshgrid(x, y)
        tx = tx.reshape((-1, 1))
        ty = ty.reshape((-1, 1))
        self.obs_rad = 2.0
        self.obs_rad2 = self.obs_rad * self.obs_rad

        self.target_x = np.stack((tx, ty), axis=1).reshape((-1, 2))

        self.target_unobserved = np.ones((self.n_agents * self.n_agents, 2), dtype=np.bool)

        # rendering initialization
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.action_scalar = 10.0

        self.seed()

    def reset(self):
        x = np.zeros((self.n_agents, self.nx_system))
        self.target_unobserved = np.ones((self.n_agents * self.n_agents, 2), dtype=np.bool)

        x[:, 0] = np.random.uniform(low=-self.px_max, high=self.px_max, size=(self.n_agents,))
        x[:, 1] = np.random.uniform(low=-self.py_max, high=self.py_max, size=(self.n_agents,))

        #bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) #+ bias[0]
        x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) #+ bias[1]

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x
        # self.a_net = self.get_connectivity(self.x)
        self.compute_helpers()
        return self.state_values, self.state_network

    def params_from_cfg(self, args):
        # TODO
        pass
        # # self.comm_radius = args.getfloat('comm_radius')
        # # self.comm_radius2 = self.comm_radius * self.comm_radius
        # # self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
        # #
        # # self.n_agents = args.getint('n_agents')
        # # self.r_max = self.r_max * np.sqrt(self.n_agents)
        #
        # # self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
        # #                                dtype=np.float32)
        # #
        # # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
        # #                                     dtype=np.float32)
        #
        # self.v_max = args.getfloat('v_max')
        # self.v_bias = self.v_max
        # self.dt = args.getfloat('dt')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        # u = np.reshape(u, (-1, 2))
        assert u.shape == (self.n_agents, self.nu)
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u * self.action_scalar

        old_x = np.copy(self.x)

        # x position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5
        # y position
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt

        # clip velocities
        self.x[:, 2:4] = np.clip(self.x[:, 2:4], -1.0*self.v_max, self.v_max)

        dist_traveled = np.sum(np.linalg.norm(self.x[:, 0:2] - old_x[:, 0:2], axis=1))

        self.compute_helpers()
        done = (0 == np.sum(self.target_unobserved))

        return (self.state_values, self.state_network), 10.0 * self.reward - dist_traveled, done, {}

    def compute_helpers(self):

        # TODO - check this, and initialize stuff in the init(), and try to make more efficient

        # Neighbors computations
        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape(
            (1, self.n_agents, self.nx_system))
        self.r2 = np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1],
                                                                                    self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

        nearest = np.argsort(self.r2, axis=1)
        obs_neigh = np.zeros((self.n_agents, self.nearest_agents * 4))
        self.adj_mat = np.zeros((self.n_agents, self.n_agents))
        for i in range(self.nearest_agents):
            ind2, ind3 = np.meshgrid(nearest[:, i], range(4), indexing='ij')
            ind1, _ = np.meshgrid(range(self.n_agents), range(4), indexing='ij')
            obs_neigh[:, i * self.nx_system:(i + 1) * self.nx_system] = np.reshape(
                self.diff[ind1.flatten(), ind2.flatten(), ind3.flatten()], (-1, 4))
            self.adj_mat[:, nearest[:, i]] = 1.0

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents, 1))  # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors

        # Targets computations
        self.diff_targets = self.x[:, 0:2].reshape((self.n_agents, 1, 2)) - self.target_x[
            self.target_unobserved].reshape(
            (1, -1, 2))
        self.r2_targets = np.multiply(self.diff_targets[:, :, 0], self.diff_targets[:, :, 0]) + np.multiply(
            self.diff_targets[:, :, 1],
            self.diff_targets[:, :, 1])

        nearest_targets = np.argsort(self.r2_targets, axis=1)
        obs_target = np.zeros((self.n_agents, self.nearest_targets * 2))

        for i in range(min(self.nearest_targets, np.shape(nearest_targets)[1])):

            ind2, ind3 = np.meshgrid(nearest_targets[:, i], range(2), indexing='ij')
            ind1, _ = np.meshgrid(range(self.n_agents), range(2), indexing='ij')
            obs_target[:, i * 2:(i + 1) * 2] = np.reshape(
                self.diff_targets[ind1.flatten(), ind2.flatten(), ind3.flatten()], (-1, 2))

        self.target_observed = np.any(self.r2_targets < self.obs_rad2, axis=0).reshape((-1, 1))
        self.target_unobserved[self.target_unobserved] = np.tile(np.logical_not(self.target_observed), (1, 2)).flatten()

        self.reward = np.sum(self.target_observed.astype(np.int))
        self.state_values = np.hstack((obs_neigh, obs_target))

        self.greedy_action = -1.0 * obs_target[:, 0:2]

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat

    def controller(self):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """

        # TODO
        # return np.zeros((self.n_agents, 2))
        return self.greedy_action / 10.0

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1], 'bo')
            locs = self.target_x[self.target_unobserved].reshape((-1, 2))
            line2, = self.ax.plot(locs[:, 0], locs[:, 1], 'rx')
            plt.ylim(-1.0 * self.py_max, 1.0 * self.py_max)
            plt.xlim(-1.0 * self.px_max, 1.0 * self.px_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1
            self.line2 = line2

            # TODO render unobserved targets
        else:
            self.line1.set_xdata(self.x[:, 0])
            self.line1.set_ydata(self.x[:, 1])
            locs = self.target_x[self.target_unobserved].reshape((-1,2))
            self.line2.set_xdata(locs[:, 0])
            self.line2.set_ydata(locs[:, 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
