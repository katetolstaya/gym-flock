import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
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

        # dim of state per agent - 2D position + Orientation
        self.nx = 3
        # number of actions per agent
        self.nu = 2

        # number of sheep and shepherds
        self.n_sheep = 10
        self.n_shepherds = 5
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
        self.force_weights = 0.1 * np.hstack(
            (4.5 * np.ones((1, self.n_shepherds, 1)), 1.0 * np.ones((1, self.n_sheep, 1))))

        # initialize state matrix
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
        self.ax = None

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

        """
        Holonomic Model
        """
        # x position
        # self.x[:, 0] = self.x[:, 0] + u[:, 0] * self.dt
        # y position
        # self.x[:, 1] = self.x[:, 1] + u[:, 1] * self.dt

        """
        Unycicle Model
        """
        # Feedback linearization
        d = 0.5  # Offset from origin
        v = u[:, 0] * np.cos(self.x[:, 2]) + u[:, 1] * np.sin(self.x[:, 2])
        w = u[:, 0] * (-np.sin(self.x[:, 2]) / d) + u[:, 1] * (np.cos(self.x[:, 2]) / d)

        # State Update (x, y, theta)
        self.x[:, 0] = self.x[:, 0] + v * np.cos(self.x[:, 2]) * self.dt
        self.x[:, 1] = self.x[:, 1] + v * np.sin(self.x[:, 2]) * self.dt
        self.x[:, 2] = self.x[:, 2] + w * self.dt

        return (self._compute_observations(), self._compute_adj_mat()), self._instant_cost(), False, {}

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

        adj_mat = (r2 < self.comm_radius_2).astype(float)

        if weighted_graph:
            np.fill_diagonal(r2, np.Inf)
            adj_mat = adj_mat / np.sqrt(r2)

        if normalize_by_neighbors:
            n_neighbors = np.reshape(np.sum(adj_mat, axis=1), (self.n_agents, 1))
            n_neighbors[n_neighbors == 0] = 1
            adj_mat = adj_mat / n_neighbors
        return adj_mat

    def _compute_sheep_controller(self):
        """
        Compute the controller for sheep. Sheep are repelled by shepherds and other sheep. Shepherd-sheep repulsion
        force is 4.5x, sheep-sheep repulsion is 1x and this is stored in self.force_weights.
        The x and y components of pairwise repulsion between agents are (delta_x / r^2, delta_y / r^2)
        where delta_x and delta_y are the relative position of one agent to another.
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
            self.ax = fig.add_subplot(111, aspect='equal')

            # plot shepherds and sheep, location and orientation using quiver
            uv = [np.cos(self.x[:, 2]), np.sin(self.x[:, 2])]
            line1 = self.ax.quiver(self.x[0:self.n_shepherds, 0], self.x[0:self.n_shepherds, 1], uv[0], uv[1], units='xy', scale=2, width=0.1, color='g', headlength=4.5, headwidth=3)
            line2 = self.ax.quiver(self.x[self.n_shepherds:, 0], self.x[self.n_shepherds:, 1], uv[0], uv[1], units='xy', scale=2, width=0.1, color='r', headlength=4.5, headwidth=3)

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
        uv = [np.cos(self.x[:, 2]), np.sin(self.x[:, 2])]
        self.line1.set_offsets(self.x[:self.n_shepherds, 0:2])
        self.line1.set_UVC(uv[0][:self.n_shepherds], uv[1][:self.n_shepherds])

        # update sheep plot
        self.line2.set_offsets(self.x[self.n_shepherds:, 0:2])
        self.line2.set_UVC(uv[0][self.n_shepherds:], uv[1][self.n_shepherds:])

        # draw updated figure
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        """
        Close the environment
        """
        pass

