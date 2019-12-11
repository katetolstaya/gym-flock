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
        """Initialize the mapping environment
        """
        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not

        # dim of state per agent, 2D position and 2D velocity
        self.nx = 4

        # agent dynamics are controlled with 2D acceleration
        self.nu = 2

        # number of robots and targets
        self.n_targets = 900
        self.n_targets_side = int(np.sqrt(self.n_targets))
        self.n_robots = 25

        # dynamics parameters
        self.dt = 0.1
        self.ddt = self.dt / 10.0
        self.v_max = 5.0  # max velocity
        self.a_max = 1  # max acceleration
        self.action_gain = 10.0  # controller gain

        # initialization parameters
        # agents are initialized uniformly at random in square r_max x r_max
        self.r_max_init = 2.0

        # graph parameters
        self.comm_radius = 5.0

        # call helper function to initialize arrays
        self.system_changed = True
        self._initialization_helper()

        # plotting and seeding parameters
        self.fig = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.np_random = None
        self.seed()

    def seed(self, seed=None):
        """ Seed the numpy random number generator
        :param seed: random seed
        :return: random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        """ Simulate a single step of the environment dynamics
        The output is (observations, adjacency matrix), cost, done_flag, options
        The observations are of dimension (Number of robots + Number of targets) x Number of observations
        Adjacency matrix is (Number of robots + Number of targets) x (Number of robots + Number of targets)
        :param u: control input for robots
        :return: described above
        """

        # TODO convert to discrete actions!
        # action will be the index of the neighbor in the graph (global index, not local)
        u = np.reshape(u, (-1, 1))
        _, diff = self._compute_inter_agent_dist_sq()
        u = -1.0 * diff[np.reshape(range(self.n_robots), (-1, 1)), u, 0:2].reshape((self.n_robots, 2))

        assert u.shape == (self.n_robots, self.nu)
        u = np.clip(u, a_min=-self.a_max, a_max=self.a_max)
        u = u * self.action_gain

        for _ in range(10):
            # position
            self.x[:self.n_robots, 0:2] = self.x[:self.n_robots, 0:2] + self.x[:self.n_robots, 2:4] * self.ddt \
                                          + u[:, 0:2] * self.ddt * self.ddt * 0.5
            # velocity
            self.x[:self.n_robots, 2:4] = self.x[:self.n_robots, 2:4] + u[:, 0:2] * self.ddt

            # clip velocity
            self.x[:self.n_robots, 2:4] = np.clip(self.x[:self.n_robots, 2:4], -self.v_max, self.v_max)

        self.system_changed = True

        # update target visitation
        adj_mat = self._compute_adj_mat(self_loops=False)
        self.visited[self.n_robots:] = np.logical_or(self.visited[self.n_robots:].flatten(),
                                                     np.any(adj_mat[self.n_robots:, 0:self.n_robots],
                                                            axis=1).flatten()).reshape((-1, 1))

        reward = np.sum(self.visited) / self.n_targets
        done = (reward == self.n_targets)

        return (self._compute_observations(), adj_mat), reward, done, {}

    def reset(self):
        """
        Reset system state. Agents are initialized in a square with side self.r_max
        :return: observations, adjacency matrix
        """
        self.x[:self.n_robots, 0:2] = self.np_random.uniform(low=-self.r_max, high=self.r_max, size=(self.n_robots, 2))
        self.x[:self.n_robots, 2:4] = self.np_random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots, 2))
        self.system_changed = True

        adj_mat = self._compute_adj_mat(self_loops=False)
        self.visited[self.n_robots:] = np.any(adj_mat[self.n_robots:, 0:self.n_robots], axis=1).flatten().reshape(
            (-1, 1))

        return self._compute_observations(), self._compute_adj_mat()

    def controller(self):
        """
        Greedy controller picks the nearest unvisited target
        :return: control action for each robot (global index of agent chosen)
        """
        r2, _ = self._compute_inter_agent_dist_sq()

        # never choose self or a visited target or another robot
        np.fill_diagonal(r2, np.Inf)
        r2[:self.n_robots, np.where(self.visited.flatten() == 1)] = np.Inf
        r2[:self.n_robots, np.where(self.agent_type.flatten() == 1)] = np.Inf

        # return the index of the closest agent
        return np.argmin(r2[:self.n_robots, :], axis=1)

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space. The robots are in green, the targets in red.
        When a target has been visited, it becomes a blue dot. The plot objects are created on the first render() call and persist between
        calls of this function.
        :param mode: required by gym
        """
        if self.fig is None:
            # initialize plot parameters
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            # plot robots and targets and visited targets as scatter plot
            line1, = self.ax.plot(self.x[0:self.n_robots, 0], self.x[0:self.n_robots, 1], 'go')
            line2, = self.ax.plot(self.x[self.n_robots:, 0], self.x[self.n_robots:, 1], 'ro')
            line3, = self.ax.plot([], [], 'b.')

            # set plot limits, axis parameters, title
            plt.ylim(-1.0 * self.r_max, self.r_max)
            plt.xlim(-1.0 * self.r_max, self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')

            # store plot state
            self.fig = fig
            self.line1 = line1
            self.line2 = line2
            self.line3 = line3

        # update robot plot
        self.line1.set_xdata(self.x[0:self.n_robots, 0])
        self.line1.set_ydata(self.x[0:self.n_robots, 1])
        temp = np.where((self.visited[self.n_robots:] == 0).flatten())

        # update unvisited target plot
        self.line2.set_xdata(self.x[self.n_robots:, 0][temp])
        self.line2.set_ydata(self.x[self.n_robots:, 1][temp])

        # update visited target plot
        self.line3.set_xdata(self.x[np.nonzero(self.visited.flatten()), 0])
        self.line3.set_ydata(self.x[np.nonzero(self.visited.flatten()), 1])

        # draw updated figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        Close the environment
        """
        pass

    def _compute_observations(self):
        """
        Uses current system state to compute the observations of agents
        The observations are the 2D positions & velocities (0 for targets) of all agents
        And the identities of the agents, 1 if robot, 0 if target
        And whether the target has been visited (1) or not (0)
        And the agent ID, 0 to (number robots + number targets)
        The dimension is (Number robots + Number targets) x 7
        :return: Observations of system state
        """
        return np.hstack((self.x, self.agent_type, self.visited, self.agent_ids))

    def _compute_inter_agent_dist_sq(self):
        """
        Compute the relative positions & velocities between all pairs of agents, and the distance between agents squared
        :return: relative position, distance squared
        """
        # TODO targets are static, don't need to recompute their distances every time, only once when initialized
        if self.system_changed:
            diff = self.x.reshape((self.n_agents, 1, self.nx)) - self.x.reshape(
                (1, self.n_agents, self.nx))
            self.r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1], diff[:, :, 1])
            self.system_changed = False
            self.diff = diff
        return self.r2, self.diff

    def _compute_adj_mat(self, weighted_graph=True, self_loops=False, normalize_by_neighbors=False):
        """
        Compute the adjacency matrix among all agents in the flock. The communication radius is fixed among all agents
        to be self.comm_radius.
        :param weighted_graph: should the graph be weighted by 1/distance to neighbors?
        :param self_loops: should self loops be present in the graph? Determines the diagonal values in the adj mat
        :param normalize_by_neighbors: should the adjacency matrix be normalized by the number of neighbors?
        :return: The adjacency matrix (Number of shepherds + Number of sheep) x (Number of shepherds + Number of sheep)
        """

        r2, _ = self._compute_inter_agent_dist_sq()
        if not self_loops:
            np.fill_diagonal(r2, np.Inf)

        adj_mat = (r2 < self.comm_radius2).astype(float)

        if weighted_graph:
            np.fill_diagonal(r2, np.Inf)
            adj_mat = adj_mat / np.sqrt(r2)

        if normalize_by_neighbors:
            n_neighbors = np.reshape(np.sum(adj_mat, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            adj_mat = adj_mat / n_neighbors
        return adj_mat

    def params_from_cfg(self, args):
        """
        Load experiment parameters from a configparser object
        :param args: loaded configparser object
        """
        # number of robots and targets
        self.n_targets = args.getint('n_targets')
        self.n_targets_side = int(np.sqrt(self.n_targets))
        self.n_robots = args.getint('n_robots')

        # load graph parameters
        self.comm_radius = args.getfloat('comm_radius')

        # load dynamics parameters
        self.v_max = args.getfloat('v_max')
        self.dt = args.getfloat('dt')
        self.ddt = self.dt / 10.0

        self._initialization_helper()

    def _initialization_helper(self):

        # number of agents
        self.n_agents = self.n_targets + self.n_robots
        self.agent_type = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))

        # initial condition
        self.r_max = self.r_max_init * np.sqrt(self.n_agents)

        # communication radius squared
        self.comm_radius2 = self.comm_radius * self.comm_radius

        # initialize state matrices
        self.x = np.zeros((self.n_agents, self.nx))
        self.visited = np.zeros((self.n_agents, 1))
        self.agent_ids = np.reshape((range(self.n_agents)), (-1, 1))

        # caching distance computation
        self.diff = np.zeros((self.n_agents, self.n_agents, self.nx))
        self.r2 = np.zeros((self.n_agents, self.n_agents))
        self.system_changed = True

        # initialize fixed grid of targets
        tempx = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tempy = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tx, ty = np.meshgrid(tempx, tempy)
        self.x[self.n_robots:, 0] = tx.flatten()
        self.x[self.n_robots:, 1] = ty.flatten()

        # problem's observation and action spaces

        # each robot picks which neighbor to move to
        self.action_space = spaces.Discrete(self.n_robots * self.n_agents)

        # see _compute_observations(self) for description of observation space
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx + 3),
                                            dtype=np.float32)
