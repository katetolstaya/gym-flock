import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
import matplotlib.patches as patches

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class ShepherdingEnv(gym.Env):

    def __init__(self):
        """Initialize the shepherding environment
        """
        self.mean_pooling = True  # normalize the adjacency matrix by the number of neighbors or not

        # dim of state per agent - 2D position
        self.nx = 2
        # number of actions per agent
        self.nu = 2

        # number of sheep and shepherds
        self.n_sheep = 50
        self.n_shepherds = 25
        self.n_agents = self.n_sheep + self.n_shepherds
        self.agent_identities = np.vstack((np.ones((self.n_shepherds, 1)), np.zeros((self.n_sheep, 1))))

        # dynamics parameters - TODO tune these parameters
        self.dt = 0.01
        self.v_max = 2.0
        self.action_scalar = 10.0  # shepherd controller gain

        # initialization parameters
        self.r_max_init = 1.0
        self.r_max = self.r_max_init * np.sqrt(self.n_agents)  # radius of disk on which agents are initialized

        # goal parameters
        self.goal_offset = np.array([-self.r_max * 3, 0])
        self.goal_region_radius = 0.5 * self.r_max

        # graph parameters
        self.comm_radius = 2.0
        self.comm_radius_2 = self.comm_radius * self.comm_radius

        # shepherd-sheep repulsion force is 4.5x, sheep-sheep repulsion is 1x  # TODO tune this
        self.force_weights = 0.1 * np.hstack((4.5 * np.ones((1, self.n_shepherds, 1)), 1.0 * np.ones((1, self.n_sheep, 1))))

        # intitialize state matrix
        self.x = np.zeros((self.n_agents, self.nx))

        # problems's observation and action spaces
        self.action_space = spaces.Box(low=-self.v_max, high=self.v_max, shape=(self.n_shepherds, self.nu),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx),
                                            dtype=np.float32)

        # plotting parameters
        self.fig = None
        self.line1 = None
        self.line2 = None

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
        The observations are of dimension (Number of shepherds + Number of sheep) x Number of observations
        Adjacency matrix is (Number of shepherds + Number of sheep) x (Number of shepherds + Number of sheep)
        :param u: control input for shepherds
        :return: described above
        """
        assert u.shape == (self.n_shepherds, self.nu)
        u = np.vstack((u * self.action_scalar, self._compute_sheep_controller()))

        # clip the velocities
        u = np.clip(u, a_min=-self.v_max, a_max=self.v_max)

        # x position
        self.x[:, 0] = self.x[:, 0] + u[:, 0] * self.dt
        # y position
        self.x[:, 1] = self.x[:, 1] + u[:, 1] * self.dt

        return (self._compute_observations(), self._compute_adj_mat()), self._instant_cost, False, {}

    def _compute_observations(self):
        """
        Uses current system state to compute the observations of agents
        The observations are the 2D positions of all agents with respect to the goal
        And the identities of the agents, 1 if shepherd, 0 if sheep
        The dimension is (Number of shepherds + Number of sheep) x 3
        :return: Observations of system state
        """
        return np.hstack((self.x, self.agent_identities))

    def _compute_inter_agent_dist_sq(self):
        """
        Compute the relative positions between all pairs of agents, and the distance between agents squared
        :return: relative position, distance squared
        """
        diff = self.x.reshape((self.n_agents, 1, self.nx)) - self.x.reshape(
            (1, self.n_agents, self.nx))
        r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1], diff[:, :, 1])
        return r2, diff

    def _compute_adj_mat(self, self_loops=False, normalize_by_neighbors=False):
        """
        Compute the adjacency matrix among all agents in the flock. The communication radius is fixed among all agents
        to be self.comm_radius
        :param self_loops: should self loops be present? Determines the diagonal values in the adj mat
        :param normalize_by_neighbors: Should the adjacency matrix be normalized by the number of neighbors?
        :return: The adjacency matrix (Number of shepherds + Number of sheep) x (Number of shepherds + Number of sheep)
        """
        r2, _ = self._compute_inter_agent_dist_sq()
        if not self_loops:
            np.fill_diagonal(r2, np.Inf)

        adj_mat = (r2 < self.comm_radius_2).astype(float)
        if normalize_by_neighbors:
            n_neighbors = np.reshape(np.sum(adj_mat, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            adj_mat = adj_mat / n_neighbors
        return adj_mat

    def _compute_sheep_controller(self):
        """
        Compute the controller for sheep. Sheep are repelled by shepherds and other sheep. Shepherd-sheep repulsion
        force is 4.5x, sheep-sheep repulsion is 1x and this is stored in self.force_weights
        :return: sheep repulsion velocities
        """
        r2, diff = self._compute_inter_agent_dist_sq()
        np.fill_diagonal(r2, np.Inf)
        potential_components = np.dstack((np.divide(diff[:, :, 0], r2), np.divide(diff[:, :, 1], r2)))
        repulsion = np.sum(self.force_weights * potential_components, axis=1)
        repulsion = repulsion.reshape((self.n_agents, self.nu))
        return repulsion[self.n_shepherds:, 0:2]

    def _instant_cost(self):
        """
        Compute the reward for the MDP, which is the fraction of sheep in the goal region
        :return: reward
        """
        return np.sum(np.linalg.norm(self.x[self.n_shepherds:, 0:2], axis=1) < self.goal_region_radius) / self.n_sheep

    def reset(self):
        """
        Reset system state. Agents are initialized on a disk of radius self.r_max
        :return: observations, adjacency matrix
        """
        # initialize agents on a disk
        length = np.sqrt(self.np_random.uniform(0, self.r_max, size=(self.n_agents,)))
        angle = np.pi * self.np_random.uniform(0, 2, size=(self.n_agents,))
        self.x[:, 0] = length * np.cos(angle)
        self.x[:, 1] = length * np.sin(angle)

        # goal is at (0, 0) and agents start at an offset from the goal
        self.x[:, 0] += self.goal_offset[0]
        self.x[:, 1] += self.goal_offset[1]

        return self._compute_observations(), self._compute_adj_mat()

    def controller(self):
        """
        Compute a baseline shepherd controller based on the potential function based approach
        :return: shepherd velocities
        """
        # TODO shepherd controller code here
        return np.zeros((self.n_shepherds, 2))

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space. The shepherds are in green, the sheep in red.
        The goal region is a red circle. The plot objects are created on the first render() call and persist between
        calls of this function. 
        :param mode: required by gym
        """
        if self.fig is None:
            # initialize plot parameters
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            # plot shepherds and sheep using scatter plot
            line1, = self.ax.plot(self.x[0:self.n_shepherds, 0], self.x[0:self.n_shepherds, 1], 'go')  # shepherds
            line2, = self.ax.plot(self.x[self.n_shepherds:, 0], self.x[self.n_shepherds:, 1], 'ro')  # sheep

            # plot red circle for goal region
            circ = patches.Circle((0, 0), self.goal_region_radius, fill=False, edgecolor='r')
            self.ax.add_patch(circ)

            # plot origin
            self.ax.plot([0], [0], 'kx')

            # set plot limits, axis parameters, title
            plt.xlim(-1.0 * self.r_max + self.goal_offset[0], self.r_max)
            plt.ylim(-1.0 * self.r_max + self.goal_offset[1], self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')

            # store plot state
            self.fig = fig
            self.line1 = line1
            self.line2 = line2

        # update shepherd plot
        self.line1.set_xdata(self.x[0:self.n_shepherds, 0])
        self.line1.set_ydata(self.x[0:self.n_shepherds, 1])

        # update sheep plot
        self.line2.set_xdata(self.x[self.n_shepherds:, 0])
        self.line2.set_ydata(self.x[self.n_shepherds:, 1])

        # draw updated figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        Close the environment
        """
        pass


    # TODO function for loading from config file
    # def params_from_cfg(self, args):
    #
    #     self.comm_radius = args.getfloat('comm_radius')
    #     self.comm_radius2 = self.comm_radius * self.comm_radius
    #     self.vr = 1 / self.comm_radius2 + np.log(self.comm_radius2)
    #
    #     self.n_sheep = args.getint('n_sheep')
    #     self.n_shepherds = args.getint('n_shepherds')
    #     self.n_agents = self.n_sheep + self.n_shepherds
    #     self.r_max = self.r_max_init * np.sqrt(self.n_agents)
    #     self.goal_offset = np.array([self.r_max * 5, self.r_max * 5])
    #
    #     self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_shepherds, 2),
    #                                    dtype=np.float32)
    #
    #     self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx_system),
    #                                         dtype=np.float32)
    #
    #     self.v_max = args.getfloat('v_max')
    #     self.v_bias = self.v_max
    #     self.dt = args.getfloat('dt')
