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
        # agents are initialized uniformly at random in square of size r_max by r_max
        self.r_max_init = 2.0

        # graph parameters
        self.comm_radius = 5.0
        self.motion_radius = 5.0
        self.obs_radius = 5.0

        # call helper function to initialize arrays
        # self.system_changed = True
        self._initialization_helper()

        # plotting and seeding parameters
        self.fig = None
        self.ax = None
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
        The output is observations, cost, done_flag, options
        :param u: control input for robots
        :return: described above
        """

        # action will be the index of the neighbor in the graph (global index, not local)
        u = np.reshape(u, (-1, 1))
        diff = self._get_pos_diff(self.x[:self.n_robots, 0:2], self.x[:, 0:2])
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

        obs, reward, done = self._get_obs_reward()

        return obs, reward, done, {}

    def _get_obs_reward(self):
        """
        Method to retrieve observation graph, with node and edge attributes
        :return:
        nodes - node attributes
        edges - edge attributes
        senders - sender nodes for each edge
        receivers - receiver nodes for each edge
        reward - MDP reward at this step
        done - is this the last step of the episode?
        """
        # observation edges from targets to nearby robots
        obs_edges, obs_dist = self._get_graph_edges(self.obs_radius,
                                                    self.x[self.n_robots:, 0:2], self.x[:self.n_robots, 0:2])
        obs_edges = (obs_edges[0] + self.n_robots, obs_edges[1])

        # communication edges among robots
        comm_edges, comm_dist = self._get_graph_edges(self.comm_radius, self.x[:self.n_robots, 0:2])

        # motion edges between targets
        motion_edges = self.motion_edges
        motion_dist = self.motion_dist

        # update target visitation
        self.visited[obs_edges[0]] = 1
        reward = np.sum(self.visited) / self.n_targets - 1.0
        done = (reward == 0.0)

        # computation graph is symmetric for now. target <-> robot undirected edges
        senders = np.concatenate((obs_edges[0], obs_edges[1], comm_edges[0], motion_edges[0]))
        receivers = np.concatenate((obs_edges[1], obs_edges[0], comm_edges[1], motion_edges[1]))
        edges = np.concatenate((obs_dist, obs_dist, comm_dist, motion_dist))
        nodes = np.hstack((self.agent_type, self.visited))
        return (nodes, edges, senders, receivers), reward, done

    def reset(self):
        """
        Reset system state. Agents are initialized in a square with side self.r_max
        :return: observations, adjacency matrix
        """
        self.x[:self.n_robots, 0:2] = self.np_random.uniform(low=-self.r_max, high=self.r_max, size=(self.n_robots, 2))
        self.x[:self.n_robots, 2:4] = self.np_random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots, 2))
        # self.system_changed = True

        self.visited.fill(0)
        obs, _, _ = self._get_obs_reward()
        return obs

    def controller(self):
        """
        Greedy controller picks the nearest unvisited target
        :return: control action for each robot (global index of agent chosen)
        """
        r = np.linalg.norm(self.x[:self.n_robots, 0:2].reshape((self.n_robots, 1, 2))
                           - self.x[self.n_robots:, 0:2].reshape((1, self.n_targets, 2)), axis=2)
        r[:, np.where(self.visited[self.n_robots:] == 1)] = np.Inf

        # return the index of the closest target
        return np.argmin(r, axis=1) + self.n_robots

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

    @staticmethod
    def _get_graph_edges(rad, pos1, pos2=None, self_loops=False):
        diff = MappingRadEnv._get_pos_diff(pos1, pos2)
        r = np.linalg.norm(diff, axis=2)
        r[r > rad] = 0
        if not self_loops and pos2 is None:
            np.fill_diagonal(r, 0)
        edges = np.nonzero(r)
        return edges, r[edges]

    @staticmethod
    def _get_pos_diff(sender_loc, receiver_loc=None):
        n = sender_loc.shape[0]
        m = sender_loc.shape[1]
        if receiver_loc is not None:
            n2 = receiver_loc.shape[0]
            m2 = receiver_loc.shape[1]
            diff = sender_loc.reshape((n, 1, m)) - receiver_loc.reshape((1, n2, m2))
        else:
            diff = sender_loc.reshape((n, 1, m)) - sender_loc.reshape((1, n, m))
        return diff

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
        """
        Initialization code that is needed after params are re-loaded
        """
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
        # self.system_changed = True

        # initialize fixed grid of targets
        tempx = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tempy = np.linspace(-1.0 * self.r_max, self.r_max, self.n_targets_side)
        tx, ty = np.meshgrid(tempx, tempy)
        self.x[self.n_robots:, 0] = tx.flatten()
        self.x[self.n_robots:, 1] = ty.flatten()

        self.motion_edges, self.motion_dist = self._get_graph_edges(self.motion_radius, self.x[self.n_robots:, 0:2])
        self.motion_edges = (self.motion_edges[0], self.motion_edges[1] + self.n_robots)

        # problem's observation and action spaces

        # each robot picks which neighbor to move to
        self.action_space = spaces.MultiDiscrete([self.n_agents] * self.n_robots)

        # see _compute_observations(self) for description of observation space
        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.nx + 3),
                                            dtype=np.float32)
