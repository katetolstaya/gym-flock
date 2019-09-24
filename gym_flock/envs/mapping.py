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
        self.dt = 0.01  # #float(config['system_dt'])
        self.v_max = 5.0  # float(config['max_vel_init'])

        self.v_bias = self.v_max

        # intitialize state matrices
        self.x = None
        self.u = None
        self.mean_vel = None
        self.init_vel = None

        self.max_accel = 1
        # self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
        #                                dtype=np.float32)
        #
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, ),
        #                                     dtype=np.float32)

        # target initialization
        self.px_max = 50
        self.py_max = 50
        x = np.linspace(-1.0 * self.px_max, self.px_max, self.n_agents)
        y = np.linspace(-1.0 * self.py_max, self.py_max, self.n_agents)

        tx, ty = np.meshgrid(x, y)
        self.obs_rad = 1.0
        self.obs_rad2 = self.obs_rad * self.obs_rad

        self.target_x = np.stack((tx, ty), axis=1)
        self.target_unobserved = np.ones((self.n_agents * self.n_agents, 1), dtype=np.bool)

        # rendering initialization
        self.fig = None
        self.line1 = None
        self.action_scalar = 10.0

        self.seed()

    def reset(self):
        x = np.zeros((self.n_agents, self.nx_system))
        self.target_unobserved = np.ones((self.n_agents * self.n_agents, 1), dtype=np.bool)

        x[:, 0] = np.random.uniform(low=-self.px_max, high=self.px_max, size=(self.n_agents,))
        x[:, 1] = np.random.uniform(low=-self.py_max, high=self.py_max, size=(self.n_agents,))

        bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0]
        x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1]

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x
        # self.a_net = self.get_connectivity(self.x)
        self.compute_helpers()
        return self.state_values, self.state_network

    def params_from_cfg(self, args):
        # TODO
        # self.comm_radius = args.getfloat('comm_radius')
        # self.comm_radius2 = self.comm_radius * self.comm_radius
        # self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
        #
        # self.n_agents = args.getint('n_agents')
        # self.r_max = self.r_max * np.sqrt(self.n_agents)

        # self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
        #                                dtype=np.float32)
        #
        # self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
        #                                     dtype=np.float32)

        self.v_max = args.getfloat('v_max')
        self.v_bias = self.v_max
        self.dt = args.getfloat('dt')

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        # u = np.reshape(u, (-1, 2))
        assert u.shape == (self.n_agents, self.nu)
        # u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u * self.action_scalar

        # x position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt + self.u[:, 0] * self.dt * self.dt * 0.5
        # y position
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt + self.u[:, 1] * self.dt * self.dt * 0.5
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.u[:, 0] * self.dt
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.u[:, 1] * self.dt

        # TODO: check distance to

        self.compute_helpers()

        return (self.state_values, self.state_network), self.reward, False, {}

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
            print(np.shape(self.diff[:, nearest[:, i], :]))
            obs_neigh[:, i*self.nx_system:(i+1)*self.nx_system] = np.reshape(self.diff[:, nearest[:, i], :], (-1, 4))
            self.adj_mat[:, nearest[:, i]] = 1.0

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents, 1))  # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors

        # Targets computations
        self.diff_targets = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.target_x[self.target_unobserved, :].reshape(
            (1, -1, 2))
        self.r2_targets = np.multiply(self.diff_targets[:, :, 0], self.diff_targets[:, :, 0]) + np.multiply(self.diff_targets[:, :, 1],
                                                                                    self.diff_targets[:, :, 1])

        nearest_targets = np.argsort(self.r2_targets, axis=1)
        obs_target = np.zeros((self.n_agents, self.nearest_targets * 2))
        for i in range(min(self.nearest_targets, np.sum(self.target_unobserved))):
            obs_target[:, i*2:(i+1)*2] = np.reshape(self.diff_targets[:, nearest_targets[:, i], :], (-1, 2))

        self.target_observed = np.any(self.r2_targets < self.obs_rad2, axis=1)
        self.target_unobserved[self.target_unobserved] = np.logical_not(self.target_observed)

        self.reward = np.sum(self.target_observed.astype(np.int))

        self.state_values = np.stack((obs_neigh, obs_target), axis=1)

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat


    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """

        # TODO
        return np.zeros((self.n_agents, 2))


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[:, 0], self.x[:, 1],
                                  'bo')  # Returns a tuple of line objects, thus the comma
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.py_max, 1.0 * self.py_max)
            plt.xlim(-1.0 * self.px_max, 1.0 * self.px_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
