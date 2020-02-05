import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from collections import OrderedDict
from gym.spaces import Box

from gym_flock.envs.spatial.make_map import generate_lattice, reject_collisions

try:
    import tensorflow as tf
except ImportError:
    tf = None


font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

N_TARGETS = 123
N_ROBOTS = 1
N_ACTIONS = 4
MAX_EDGES = 5
N_ACTIVE_TARGETS = 60

# N_TARGETS = 36
# N_ROBOTS = 1
# N_ACTIONS = 2
# MAX_EDGES = 10
# N_ACTIVE_TARGETS = 10
# GRID = False
# SQUARE = True

# N_TARGETS = 900
# N_ROBOTS = 10
# N_ACTIONS = 15
# MAX_EDGES = 15
# N_ACTIVE_TARGETS = 200
# GRID = True

CIRCLES = True

# GRID = 0
# SQUARE = 1
# SPARSE_GRID = 2
#
# MAP_TYPE = GRID


class MappingRadEnv(gym.Env):
    def __init__(self):
        """Initialize the mapping environment
        """
        super(MappingRadEnv, self).__init__()
        self.np_random = None
        self.seed()

        # dim of state per agent, 2D position and 2D velocity
        self.nx = 4
        self.velocity_control = True

        # agent dynamics are controlled with 2D acceleration
        self.nu = 2

        # number of robots and targets
        self.n_targets = N_TARGETS
        # self.n_targets_side = int(np.sqrt(self.n_targets))
        self.n_robots = N_ROBOTS

        # dynamics parameters
        self.dt = 2.0
        self.ddt = self.dt / 10.0
        self.v_max = 3.0  # max velocity
        self.a_max = 3.0  # max acceleration
        self.action_gain = 1.0  # controller gain

        # initialization parameters
        # agents are initialized uniformly at random in square of size r_max by r_max
        self.r_max_init = 2.0
        self.x_max_init = 2.0
        self.y_max_init = 2.0

        # graph parameters
        self.comm_radius = 6.0
        self.motion_radius = 6.0
        self.obs_radius = 6.0

        # call helper function to initialize arrays
        # self.system_changed = True
        self._initialization_helper()

        # plotting and seeding parameters
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.line3 = None

    def seed(self, seed=None):
        """ Seed the numpy random number generator
        :param seed: random seed
        :return: random seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u_ind):
        """ Simulate a single step of the environment dynamics
        The output is observations, cost, done_flag, options
        :param u_ind: control input for robots
        :return: described above
        """

        # action will be the index of the neighbor in the graph
        u_ind = np.reshape(u_ind, (-1, 1))
        robots_index = np.reshape(range(self.n_robots), (-1, 1))
        u_ind = np.reshape(self.mov_edges[1], (self.n_robots, self.n_actions))[robots_index, u_ind]

        # # project onto the nearest 5 edges
        # n_nearby_nodes = 10
        # mov_edges, _ = self._get_k_edges(n_nearby_nodes, self.x[:self.n_robots, 0:2], self.x[self.n_robots:, 0:2])
        # r = np.linalg.norm(self.x[u_ind, 0:2].reshape((self.n_robots, 1, 2))
        #                    - self.x[:, 0:2].reshape((1, self.n_agents, 2)), axis=2)
        # u_ind = np.argmin(np.reshape(r[mov_edges], (N_ROBOTS, n_nearby_nodes)), axis=1)

        for _ in range(10):
            diff = self._get_pos_diff(self.x[:self.n_robots, 0:2], self.x[:, 0:2])
            u = -1.0 * diff[robots_index, u_ind, 0:2].reshape((self.n_robots, 2))
            u = np.clip(u, a_min=-self.a_max, a_max=self.a_max)
            u = (u + 0.1 * (self.np_random.uniform(size=(self.n_robots, 2)) - 0.5)) * self.action_gain

            if self.velocity_control:
                self.x[:self.n_robots, 0:2] = self.x[:self.n_robots, 0:2] + u[:, 0:2] * self.ddt
            else:
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
        # movement edges from robots to K random landmarks
        # mov_edges, mov_dist = self._get_k_random_edges(self.np_random, self.n_actions, self.x[:self.n_robots, 0:2], self.x[self.n_robots:, 0:2])
        mov_edges, mov_dist = self._get_k_edges(self.n_actions, self.x[:self.n_robots, 0:2], self.x[self.n_robots:, 0:2])
        mov_edges = (mov_edges[0], mov_edges[1] + self.n_robots)
        self.mov_edges = mov_edges
        # print(len(mov_edges[0]))
        # print(N_ACTIONS)
        # print(self.n_robots)
        assert len(mov_edges[0]) == N_ACTIONS * self.n_robots

        # communication edges among robots
        comm_edges, comm_dist = self._get_graph_edges(self.comm_radius, self.x[:self.n_robots, 0:2])

        # observation edges from robots to nearby landmarks
        obs_edges, obs_dist = self._get_graph_edges(self.motion_radius, self.x[:self.n_robots, 0:2],  self.x[self.n_robots:, 0:2])
        obs_edges = (obs_edges[0], obs_edges[1] + self.n_robots)

        # sensor edges from targets to nearby robots
        # sensor_edges, _ = self._get_graph_edges(self.obs_radius,
        #                                             self.x[self.n_robots:, 0:2], self.x[:self.n_robots, 0:2])
        # sensor_edges[0] += self.n_robots
        # update target visitation

        self.visited[obs_edges[1]] = 1
        reward = np.sum(self.visited[self.n_robots:]) - self.n_targets
        done = np.sum(self.visited[self.n_robots:]) == self.n_targets

        # we want to fix the number of edges into the robot from targets.
        senders = np.concatenate((obs_edges[0], mov_edges[1], comm_edges[0], self.motion_edges[0]))
        receivers = np.concatenate((obs_edges[1], mov_edges[0], comm_edges[1], self.motion_edges[1]))
        edges = np.concatenate((obs_dist, mov_dist, comm_dist, self.motion_dist)).reshape((-1, 1))

        # edges = 1.0/(edges + 0.1)
        # edges.fill(1.0)

        # -1 indicates unused edges
        self.senders.fill(-1)
        self.receivers.fill(-1)

        assert len(senders) <= np.shape(self.senders)[0]

        self.senders[:len(senders)] = senders
        self.receivers[:len(receivers)] = receivers

        self.edges[:edges.shape[0], :edges.shape[1]] = edges
        self.nodes[:, 0] = self.agent_type.flatten()
        self.nodes[:, 1] = np.logical_not(self.visited).flatten()

        obs = {'nodes': self.nodes, 'edges': self.edges, 'senders': self.senders, 'receivers': self.receivers}

        return obs, reward, done

    def get_visitation(self, pos, visited):
        # observation edges from robots to nearby landmarks
        obs_edges, obs_dist = self._get_graph_edges(self.motion_radius, pos[:self.n_robots, 0:2],  pos[self.n_robots:, 0:2])
        obs_edges = (obs_edges[0], obs_edges[1] + self.n_robots)
        visited[obs_edges[1]] = 1
        return visited

    def reset(self):
        """
        Reset system state. Agents are initialized in a square with side self.r_max
        :return: observations, adjacency matrix
        """
        self.x[:self.n_robots, 2:4] = self.np_random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots, 2))

        # initialize robots near targets
        self.x[:self.n_robots, 0:2] = self.x[self.np_random.choice(self.n_targets, size=(self.n_robots,))+self.n_robots,0:2]
        self.x[:self.n_robots, 0:2] += self.np_random.uniform(low=-0.5*self.motion_radius, high=0.5*self.motion_radius, size=(self.n_robots, 2))

        self.visited.fill(1)
        self.visited[self.np_random.choice(self.n_targets, size=(N_ACTIVE_TARGETS,), replace=False)+self.n_robots] = 0

        obs, _, _ = self._get_obs_reward()
        return obs

    def controller(self, random=False):
        """
        Greedy controller picks the nearest unvisited target
        :return: control action for each robot (global index of agent chosen)
        """
        if not random:
            # get closest unvisited
            r = np.linalg.norm(self.x[:self.n_robots, 0:2].reshape((self.n_robots, 1, 2))
                               - self.x[self.n_robots:, 0:2].reshape((1, self.n_targets, 2)), axis=2)
            r[:, np.where(self.visited[self.n_robots:] == 1)] = np.Inf

            # get the closest neighbor to the unvisited target
            min_unvisited = np.argmin(r, axis=1) + self.n_robots
            r = np.linalg.norm(self.x[min_unvisited, 0:2].reshape((self.n_robots, 1, 2))
                               - self.x[:, 0:2].reshape((1, self.n_agents, 2)), axis=2)
            action = np.argmin(np.reshape(r[self.mov_edges], (self.n_robots, N_ACTIONS)), axis=1)
            return action
        else:
            return self.np_random.choice(self.n_actions, size=(self.n_robots, 1))

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space. The robots are in green, the targets in red.
        When a target has been visited, it becomes a blue dot. The plot objects are created on the first render() call and persist between
        calls of this function.
        :param mode: 'human' mode renders the environment, and nothing happens otherwise
        """
        if mode is not 'human':
            return

        if self.fig is None:
            # initialize plot parameters
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            # plot robots and targets and visited targets as scatter plot
            line1, = self.ax.plot(self.x[0:self.n_robots, 0], self.x[0:self.n_robots, 1], 'go')
            line2, = self.ax.plot(self.x[self.n_robots:, 0], self.x[self.n_robots:, 1], 'ro')
            line3, = self.ax.plot([], [], 'b.')

            if CIRCLES:
                for (x, y) in zip(self.x[self.n_robots:, 0], self.x[self.n_robots:, 1]):
                    circle = plt.Circle((x, y), radius=self.motion_radius, facecolor='none', edgecolor='k')
                    self.ax.add_patch(circle)

            # set plot limits, axis parameters, title
            plt.xlim(self.x_min, self.x_max)
            plt.ylim(self.y_min, self.y_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            # plt.title('GNN Controller')

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
        """
        Get list of edges from agents in positions pos1 to positions pos2.
        for agents closer than distance rad
        :param rad: "communication" radius
        :param pos1: first set of positions
        :param pos2: second set of positions
        :param self_loops: boolean flag indicating whether to include self loops
        :return: (senders, receivers), edge features
        """
        diff = MappingRadEnv._get_pos_diff(pos1, pos2)
        r = np.linalg.norm(diff, axis=2)
        r[r > rad] = 0
        if not self_loops and pos2 is None:
            np.fill_diagonal(r, 0)
        edges = np.nonzero(r)
        return edges, r[edges]

    @staticmethod
    def _get_pos_diff(sender_loc, receiver_loc=None):
        """
        Get matrix of distances between agents in positions pos1 to positions pos2.
        :param sender_loc: first set of positions
        :param receiver_loc: second set of positions (use sender_loc if None)
        :return: matrix of distances, len(pos1) x len(pos2)
        """
        n = sender_loc.shape[0]
        m = sender_loc.shape[1]
        if receiver_loc is not None:
            n2 = receiver_loc.shape[0]
            m2 = receiver_loc.shape[1]
            diff = sender_loc.reshape((n, 1, m)) - receiver_loc.reshape((1, n2, m2))
        else:
            diff = sender_loc.reshape((n, 1, m)) - sender_loc.reshape((1, n, m))
        return diff

    @staticmethod
    def _get_k_edges(k, pos1, pos2=None, self_loops=False):
        """
        Get list of edges from agents in positions pos1 to closest agents in positions pos2.
        Each agent in pos1 will have K outgoing edges.
        :param k: number of edges
        :param pos1: first set of positions
        :param pos2: second set of positions
        :param self_loops: boolean flag indicating whether to include self loops
        :return: (senders, receivers), edge features
        """
        diff = MappingRadEnv._get_pos_diff(pos1, pos2)
        r = np.linalg.norm(diff, axis=2)
        if not self_loops and pos2 is None:
            np.fill_diagonal(r, np.Inf)

        # idx = np.argpartition(r, k-1, axis=1)[:, 0:k]

        idx = np.argpartition(r, k, axis=1)[:, 0:k+1]
        mask = np.zeros(np.shape(r))
        mask[np.arange(np.shape(pos1)[0])[:, None], idx] = 1
        # remove the closest edge
        idx = np.argmin(r, axis=1)
        mask[np.arange(np.shape(pos1)[0])[:], idx] = 0

        r = r * mask

        edges = np.nonzero(r)
        return edges, r[edges]

    def _initialization_helper(self):
        """
        Initialization code that is needed after params are re-loaded
        """

        self.y_min = 0
        self.x_min = 0
        self.x_max = 100
        self.y_max = 100

        # triangular lattice
        lattice_vectors = [
            2.75 * np.array([-1.414, -1.414]),
            2.75 * np.array([-1.414, 1.414])]

        # # square lattice
        # lattice_vectors = [
        #     np.array([-5.5, 0.]),
        #     np.array([0., -5.5])]

        spots = generate_lattice((self.x_min, self.x_max, self.y_min, self.y_max), 2 * lattice_vectors)

        obstacles = [(10, 45, 10, 90), (55, 90, 10, 90)]
        spots = reject_collisions(spots, obstacles)

        self.n_targets = np.shape(spots)[0]
        # print(self.n_targets)
        self.n_agents = self.n_targets + self.n_robots
        self.x = np.zeros((self.n_agents, self.nx))
        self.x[self.n_robots:, 0:2] = spots

        # if MAP_TYPE == GRID:
        #     self.gen_grid()
        # elif MAP_TYPE == SQUARE:
        #     self.gen_square()
        # elif MAP_TYPE == SPARSE_GRID:
        #     self.gen_sparse_grid()

        self.max_edges = self.n_agents * MAX_EDGES
        self.agent_type = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))
        self.n_actions = N_ACTIONS

        self.edges = np.zeros((self.max_edges, 1), dtype=np.float32)
        self.nodes = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.senders = -1 * np.ones((self.max_edges,), dtype=np.int32)
        self.receivers = -1 * np.ones((self.max_edges,), dtype=np.int32)

        # communication radius squared
        self.comm_radius2 = self.comm_radius * self.comm_radius

        # initialize state matrices

        self.visited = np.ones((self.n_agents, 1))
        self.visited[self.np_random.choice(self.n_targets, size=(N_ACTIVE_TARGETS,), replace=False)+self.n_robots] = 0

        self.agent_ids = np.reshape((range(self.n_agents)), (-1, 1))

        # caching distance computation
        self.diff = np.zeros((self.n_agents, self.n_agents, self.nx))
        self.r2 = np.zeros((self.n_agents, self.n_agents))

        self.motion_edges, self.motion_dist = self._get_graph_edges(self.motion_radius, self.x[self.n_robots:, 0:2], self_loops=True)

        # self.motion_edges, self.motion_dist = self._get_k_edges(self.n_actions + 1, self.x[self.n_robots:, 0:2], self_loops=True)
        self.motion_edges = (self.motion_edges[0] + self.n_robots, self.motion_edges[1] + self.n_robots)

        # problem's observation and action spaces
        if self.n_robots == 1:
            self.action_space = spaces.Discrete(self.n_actions)
        else:
            self.action_space = spaces.MultiDiscrete([self.n_actions] * self.n_robots)

        self.observation_space = gym.spaces.Dict(
            [
                ("nodes", Box(shape=(self.n_agents, 2), low=-np.Inf, high=np.Inf, dtype=np.float32)),
                ("edges", Box(shape=(self.max_edges, 1), low=-np.Inf, high=np.Inf, dtype=np.float32)),
                ("senders", Box(shape=(self.max_edges, 1), low=0, high=self.n_agents, dtype=np.float32)),
                ("receivers", Box(shape=(self.max_edges, 1), low=0, high=self.n_agents, dtype=np.float32)),
            ]
        )

    @staticmethod
    def unpack_obs(obs):
        assert tf is not None, "Function unpack_obs() is not available if Tensorflow is not imported."
        n_nodes = N_ROBOTS + N_TARGETS
        max_edges = MAX_EDGES
        max_n_edges = n_nodes * max_edges
        dim_edges = 1
        dim_nodes = 2

        # unpack node and edge data from flattened array
        shapes = ((n_nodes, dim_nodes), (max_n_edges, dim_edges), (max_n_edges, 1), (max_n_edges, 1))
        sizes = [np.prod(s) for s in shapes]
        tensors = tf.split(obs, sizes, axis=1)
        tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
        nodes, edges, senders, receivers = tensors
        batch_size = tf.shape(nodes)[0]

        n_node = tf.fill((batch_size,), n_nodes)  # assume n nodes is fixed
        nodes = tf.reshape(nodes, (-1, dim_nodes))

        # compute edge mask and number of edges per graph
        mask = tf.reshape(tf.not_equal(senders, -1), (batch_size, -1))  # padded edges have sender = -1
        n_edge = tf.reduce_sum(tf.cast(mask, tf.float32), axis=1)
        mask = tf.reshape(mask, (-1,))

        # flatten edge data
        edges = tf.reshape(edges, (-1, dim_edges))
        senders = tf.reshape(senders, (-1,))
        receivers = tf.reshape(receivers, (-1,))

        # mask edges
        edges = tf.boolean_mask(edges, mask, axis=0)
        senders = tf.boolean_mask(senders, mask)
        receivers = tf.boolean_mask(receivers, mask)

        # cast all indices to int
        n_node = tf.cast(n_node, tf.int32)
        n_edge = tf.cast(n_edge, tf.int32)
        senders = tf.cast(senders, tf.int32)
        receivers = tf.cast(receivers, tf.int32)

        # TODO this is a hack - want global outputs, but have no global inputs
        globs = tf.fill((batch_size, 1), 0.0)

        return batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs

    def gen_square(self):
        self.x_max = self.x_max_init * self.n_agents / 4
        self.y_max = self.y_max_init * self.n_agents / 4
        per_side = int(self.n_targets / 4)

        targets = set()

        # initialize fixed grid of targets
        tempx = np.linspace(-self.x_max, -self.x_max, 1)
        tempy = np.linspace(-self.y_max, self.y_max, per_side, endpoint=False)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(self.x_max, self.x_max, 1)
        tempy = np.linspace(-self.y_max, self.y_max, per_side, endpoint=False)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(-self.x_max, self.x_max, per_side, endpoint=False)
        tempy = np.linspace(self.y_max, self.y_max, 1)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(-self.x_max, self.x_max, per_side, endpoint=False)
        tempy = np.linspace(-self.y_max, -self.y_max, 1)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))
        targets.add((self.x_max, self.y_max))

        targets = list(zip(*targets))

        self.x[self.n_robots:, 0] = targets[0]
        self.x[self.n_robots:, 1] = targets[1]

    def gen_grid(self):
        self.n_targets_side = int(np.sqrt(self.n_targets))
        self.x_max = self.x_max_init * self.n_targets_side
        self.y_max = self.y_max_init * self.n_targets_side
        tempx = np.linspace(-1.0 * self.x_max, self.x_max, self.n_targets_side)
        tempy = np.linspace(-1.0 * self.y_max, self.y_max, self.n_targets_side)
        tx, ty = np.meshgrid(tempx, tempy)
        self.x[self.n_robots:, 0] = tx.flatten()
        self.x[self.n_robots:, 1] = ty.flatten()

    def gen_sparse_grid(self):
        self.x_max = self.x_max_init * self.n_agents / 6
        self.y_max = self.y_max_init * self.n_agents / 6

        per_side = int(self.n_targets / 6)

        targets = set()

        # initialize fixed grid of targets
        tempx = np.linspace(-self.x_max, -self.x_max, 1)
        tempy = np.linspace(-self.y_max, self.y_max, per_side, endpoint=False)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(self.x_max, self.x_max, 1)
        tempy = np.linspace(-self.y_max, self.y_max, per_side, endpoint=False)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(0, 0, 1)
        tempy = np.linspace(-self.y_max + self.y_max_init, self.y_max, per_side, endpoint=False)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(-self.x_max, self.x_max, per_side, endpoint=False)
        tempy = np.linspace(self.y_max, self.y_max, 1)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(-self.x_max, self.x_max, per_side, endpoint=False)
        tempy = np.linspace(-self.y_max, -self.y_max, 1)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        tempx = np.linspace(-self.x_max + self.x_max_init, self.x_max, per_side, endpoint=False)
        tempy = np.linspace(0, 0, 1)
        tx, ty = np.meshgrid(tempx, tempy)
        targets = targets.union(set(zip(tx.flatten(), ty.flatten())))

        targets.add((self.x_max, self.y_max))

        targets = list(zip(*targets))

        self.x[self.n_robots:, 0] = targets[0]
        self.x[self.n_robots:, 1] = targets[1]