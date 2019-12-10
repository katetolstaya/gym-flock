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


class MappingRadEnv(gym.Env):

    def __init__(self):
        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not

        # dim of state per agent, 2D position plus 2D velocity
        self.nx_system = 4
        # number of actions per agent
        self.nu = 2

        # default problem parameters
        self.n_targets = 400
        self.n_targets_side = int(np.sqrt(self.n_targets))
        self.n_robots = 25
        self.n_agents = self.n_targets + self.n_robots
        self.comm_radius = 5.0  # float(config['comm_radius'])
        self.dt = 0.01  # #float(config['system_dt'])
        self.ddt = self.dt / 10.0
        self.v_max = 5.0  # float(config['max_vel_init'])
        self.r_max_init = 2.0  # 10.0  #  float(config['max_rad_init'])
        self.r_max = self.r_max_init * np.sqrt(self.n_agents)
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
        self.robot_flag = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))

        # intitialize state matrices
        self.x = None
        self.u = None
        self.adj_mat_mean = None
        self.x_features = None
        self.visited = None
        self.adj_mat = None
        self.state_network = None
        self.observations = None
        self.r2 = None
        self.diff = None
        self.np_random = None

        self.max_accel = 1
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_robots, self.nu),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx_system),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None
        self.line2 = None
        self.action_scalar = 10.0  # controller gain

        self.seed()

        self.x = np.zeros((self.n_agents, self.nx_system))
        tempx = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tempy = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tx, ty = np.meshgrid(tempx, tempy)
        self.x[self.n_robots:, 0] = tx.flatten()
        self.x[self.n_robots:, 1] = ty.flatten()

    def params_from_cfg(self, args):

        self.comm_radius = args.getfloat('comm_radius')
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)

        self.n_targets = args.getint('n_targets')
        self.n_targets_side = int(np.sqrt(self.n_targets))
        self.n_robots = args.getint('n_robots')
        self.n_agents = self.n_targets + self.n_robots
        self.r_max = self.r_max_init * np.sqrt(self.n_agents)

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_robots, 2),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx_system),
                                            dtype=np.float32)

        self.robot_flag = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))
        self.v_max = args.getfloat('v_max')
        self.dt = args.getfloat('dt')
        self.ddt = self.dt / 10.0

        self.x = np.zeros((self.n_agents, self.nx_system))
        tempx = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tempy = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tx, ty = np.meshgrid(tempx, tempy)
        self.x[self.n_robots:, 0] = tx.flatten()
        self.x[self.n_robots:, 1] = ty.flatten()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        assert u.shape == (self.n_robots, self.nu)
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u * self.action_scalar

        for _ in range(10):
            # x position
            self.x[:self.n_robots, 0] = self.x[:self.n_robots, 0] + self.x[:self.n_robots, 2] * self.ddt \
                                        + self.u[:, 0] * self.ddt * self.ddt * 0.5
            # y position
            self.x[:self.n_robots, 1] = self.x[:self.n_robots, 1] + self.x[:self.n_robots, 3] * self.ddt \
                                        + self.u[:, 1] * self.ddt * self.ddt * 0.5
            # x velocity
            self.x[:self.n_robots, 2] = np.clip(self.x[:self.n_robots, 2] + self.u[:, 0] * self.ddt, -self.v_max,
                                                self.v_max)
            # y velocity
            self.x[:self.n_robots, 3] = np.clip(self.x[:self.n_robots, 3] + self.u[:, 1] * self.ddt, -self.v_max,
                                                self.v_max)

        self.compute_helpers()

        return (self.observations, self.state_network), self.instant_cost(), False, {}

    def compute_helpers(self):

        # TODO - graph between targets doesn't change, so precompute?
        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape(
            (1, self.n_agents, self.nx_system))
        self.r2 = np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1],
                                                                                    self.diff[:, :, 1])

        self.adj_mat = (self.r2 < self.comm_radius2).astype(float)

        self.visited[self.n_robots:] = np.logical_or(self.visited[self.n_robots:].flatten(),
                                                     np.any(self.adj_mat[self.n_robots:, 0:self.n_robots], axis=1).flatten()).reshape((-1,1))

        # Normalize the adjacency matrix by the number of neighbors (mean pooling, instead of sum pooling)
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents, 1))
        self.adj_mat_mean = self.adj_mat / n_neighbors

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat

        self.observations = np.hstack((self.x, self.robot_flag, self.visited))

    def instant_cost(self):  # TODO - per agent or one scalar for team?
        return np.sum(self.visited)

    def reset(self):

        # TODO random target locations or grid?
        self.x[:self.n_robots, 0] = np.random.uniform(low=-self.r_max, high=self.r_max, size=(self.n_robots,))
        self.x[:self.n_robots, 1] = np.random.uniform(low=-self.r_max, high=self.r_max, size=(self.n_robots,))
        self.x[:self.n_robots, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots,))
        self.x[:self.n_robots, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots,))

        self.visited = np.zeros((self.n_agents, 1))
        self.compute_helpers()
        return self.observations, self.state_network

    def controller(self, centralized=None):
        return np.zeros((self.n_robots, 2))
        # TODO - return the direction of the nearest unvisited target
        # pass

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """
        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)
            line1, = self.ax.plot(self.x[0:self.n_robots, 0], self.x[0:self.n_robots, 1], 'go')
            line2, = self.ax.plot(self.x[self.n_robots:, 0], self.x[self.n_robots:, 1], 'r'
                                                                                        'o')
            line3, = self.ax.plot([], [], 'b.')
            self.ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, self.r_max)
            plt.xlim(-1.0 * self.r_max, self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1
            self.line2 = line2
            self.line3 = line3

        self.line1.set_xdata(self.x[0:self.n_robots, 0])
        self.line1.set_ydata(self.x[0:self.n_robots, 1])
        # print(np.nonzero(self.visited))
        temp = np.where((self.visited[self.n_robots:] == 0).flatten())

        self.line2.set_xdata(self.x[self.n_robots:, 0][temp])
        self.line2.set_ydata(self.x[self.n_robots:, 1][temp])

        self.line3.set_xdata(self.x[np.nonzero(self.visited.flatten()), 0])
        self.line3.set_ydata(self.x[np.nonzero(self.visited.flatten()), 1])

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
