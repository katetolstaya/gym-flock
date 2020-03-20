import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from gym.spaces import Box

from gym_flock.envs.spatial.make_map import generate_lattice, reject_collisions, gen_obstacle_grid, in_obstacle, \
    generate_geometric_roads
from gym_flock.envs.spatial.vrp_solver import solve_vrp
from gym_flock.envs.spatial.utils import _get_graph_edges, _get_k_edges, _get_pos_diff

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import ortools
    from ortools.constraint_solver import routing_enums_pb2
    from ortools.constraint_solver import pywrapcp
except ImportError:
    ortools = None

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

# number of node and edge features
# N_NODE_FEAT = 3
N_NODE_FEAT = 4
# N_EDGE_FEAT = 1
N_EDGE_FEAT = 2
N_GLOB_FEAT = 1
DECAY_COEF = 1.0
# DECAY_COEF = 0.9

COMM_EDGES = False

# padding for a variable number of graph edges
PAD_NODES = True
MAX_NODES = 700
MAX_EDGES = 3

# number of edges/actions for each robot, fixed
N_ACTIONS = 4
ALLOW_NEAREST = False
GREEDY_CONTROLLER = False
# GREEDY_CONTROLLER = True

EPISODE_LENGTH = 30
# EPISODE_LENGTH = 100
# EARLY_TERMINATION = True
EARLY_TERMINATION = False
# EPISODE_LENGTH = 30

# parameters for map generation
# ranges = [(5, 30),  (35, 65), (70, 95)]
# ranges = [(5, 25), (30, 50), (57, 75), (80, 95)]


# ranges = [(5, 65), (70, 130), (135, 195)]
ranges = [(5, 50), (55, 100), (110, 150), (160, 195)]
# ranges = [(5, 50), (55, 100), (105, 150), (155, 195),(205, 250), (260, 300), (310, 355), (360, 395)]

# OBST = gen_obstacle_grid(ranges)

OBST = []

# N_ROBOTS = 5
N_ROBOTS = 5
# XMAX = 100
# YMAX = 100
XMAX = 200
YMAX = 200
# XMAX = 400
# YMAX = 400

# FRAC_ACTIVE = 1.0
FRAC_ACTIVE = 0.5
MIN_FRAC_ACTIVE = 0.5

# unvisited_regions = [(0, 200, 200, 400), (200, 400, 0, 200)]
# start_regions = [(75, 125, 150, 200)]

unvisited_regions = [(0, 200, 0, 200)]
# unvisited_regions = [(0, 70, 60, 200), (130, 200, 0, 200)]
# start_regions = [(0, 70, 0, 70)]
#
# unvisited_regions = [(0, 30, 25, 100), (55, 100, 0, 57)]
# unvisited_regions = [(0, 35, 30, 70), (65, 100, 0, 100)]

# start_regions = [(30, 70, 30, 70)]
# start_regions = [(0, 200, 0, 200)]
# start_regions = [(0, 100, 0, 100)]
start_regions = [(50, 150, 50, 150)]


# start_regions = [(0, 70, 0, 70)]
# start_regions = [(0, 35, 0, 35)]

DELTA = 5.5


class MappingRadEnv(gym.Env):
    def __init__(self, n_robots=N_ROBOTS, frac_active_targets=FRAC_ACTIVE, obstacles=OBST, xmax=XMAX, ymax=YMAX,
                 starts=start_regions, unvisiteds=unvisited_regions):
        """Initialize the mapping environment
        """
        super(MappingRadEnv, self).__init__()

        self.episode_length = EPISODE_LENGTH

        self.y_min = -ymax/2
        self.x_min = -xmax/2
        self.x_max = xmax/2
        self.y_max = ymax/2
        self.obstacles = obstacles
        self.start_ranges = starts
        self.unvisited_ranges = unvisiteds

        # # triangular lattice
        # self.lattice_vectors = [
        #     2.75 * np.array([-1.414, -1.414]),
        #     2.75 * np.array([-1.414, 1.414])]

        # square lattice
        self.lattice_vectors = [
            np.array([-DELTA, 0.]),
            np.array([0., -DELTA])]

        self.np_random = None
        self.seed()

        # dim of state per agent, 2D position and 2D velocity
        self.nx = 2

        # agent dynamics are controlled with 2D acceleration
        self.nu = 2

        # number of robots and targets
        self.n_robots = n_robots
        self.frac_active_targets = frac_active_targets

        # graph parameters
        self.comm_radius = 20.0
        self.motion_radius = 7.0
        self.obs_radius = 7.0

        # call helper function to initialize arrays
        self._initialize_graph()

        # plotting and seeding parameters
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.cached_solution = None
        self.graph_previous = None
        self.graph_cost = None

        self.step_counter = 0
        self.n_motion_edges = 0
        self.done = False
        self.last_loc = None
        self.node_history = None

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
        self.last_loc = self.closest_targets

        next_loc = copy.copy(u_ind)
        for i in range(self.n_robots):
            next_loc[i] = self.mov_edges[1][np.where(self.mov_edges[0] == i)][u_ind[i]]

        self.x[:self.n_robots, 0:2] = self.x[next_loc.flatten(), 0:2]

        obs, reward, done = self._get_obs_reward()
        done = done or (EARLY_TERMINATION and self.done)
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
        # action edges from landmarks to robots
        action_edges, action_dist = _get_k_edges(self.n_actions, self.x[:self.n_robots, 0:2],
                                                 self.x[self.n_robots:self.n_agents, 0:2], allow_nearest=ALLOW_NEAREST)
        action_edges = (action_edges[0], action_edges[1] + self.n_robots)
        assert len(action_edges[0]) == N_ACTIONS * self.n_robots, "Number of action edges is not num robots x n_actions"
        self.mov_edges = action_edges

        # planning edges from robots to landmarks
        plan_edges, plan_dist = _get_graph_edges(1.0, self.x[:self.n_robots, 0:2], self.x[self.n_robots:self.n_agents, 0:2])
        plan_edges = (plan_edges[0], plan_edges[1] + self.n_robots)

        old_sum = np.sum(self.visited[self.n_robots:self.n_agents])
        self.visited[self.closest_targets] = 1

        if N_NODE_FEAT == 4:
            self.node_history = DECAY_COEF * self.node_history
            self.node_history[self.closest_targets] = 1

        if COMM_EDGES:
            # communication edges among robots
            comm_edges, comm_dist = _get_graph_edges(self.comm_radius, self.x[:self.n_robots, 0:2])

            senders = np.concatenate((plan_edges[0], action_edges[1], comm_edges[0]))
            receivers = np.concatenate((plan_edges[1], action_edges[0], comm_edges[1]))
            edges_dist = np.concatenate((plan_dist, action_dist, comm_dist)).reshape((-1, N_EDGE_FEAT))
        else:
            senders = np.concatenate((plan_edges[0], action_edges[1]))
            receivers = np.concatenate((plan_edges[1], action_edges[0]))
            edges_dist = np.concatenate((plan_dist, action_dist)).reshape((-1, 1))
        assert len(senders) + self.n_motion_edges <= np.shape(self.senders)[0], "Increase MAX_EDGES"

        # TODO the reciprocal of distance necessary?
        edges_dist = (DELTA + 1.0) / (edges_dist + 1.0)

        if N_EDGE_FEAT == 2:
            # TODO is the edge history necessary as a form of memory?
            last_edges = np.zeros((len(senders), 1), dtype=np.bool)
            if self.last_loc is not None:
                for i in range(self.n_robots):
                    last_edges = np.logical_or(last_edges,
                                               np.logical_and(receivers == i, senders == self.last_loc[i]).reshape(
                                                   (-1, 1)))
                    last_edges = last_edges.reshape((-1, 1))
            edges = np.hstack((last_edges, edges_dist)).reshape((-1, N_EDGE_FEAT))
        else:
            edges = edges_dist.reshape((-1, N_EDGE_FEAT))

        # -1 indicates unused edges
        self.senders[self.n_motion_edges:] = -1
        self.receivers[self.n_motion_edges:] = -1
        self.nodes.fill(0)

        self.senders[-len(senders):] = senders
        self.receivers[-len(receivers):] = receivers
        self.edges[-len(senders):, :] = edges

        self.nodes[0:self.n_agents, 0] = self.robot_flag.flatten()
        self.nodes[0:self.n_agents, 1] = self.landmark_flag.flatten()
        self.nodes[0:self.n_agents, 2] = np.logical_not(self.visited).flatten()

        if N_NODE_FEAT == 4:
            self.nodes[0:self.n_agents, 3] = self.node_history.flatten()

        step_array = np.array([self.step_counter]).reshape((1, 1))

        obs = {'nodes': self.nodes, 'edges': self.edges, 'senders': self.senders, 'receivers': self.receivers,
               'step': step_array}

        self.step_counter += 1
        done = self.step_counter == self.episode_length or np.sum(self.visited[self.n_robots:]) == self.n_targets

        reward = np.sum(self.visited[self.n_robots:]) - old_sum
        return obs, reward, done

    def reset(self):
        """
        Reset system state. Agents are initialized in a square with side self.r_max
        :return: observations, adjacency matrix
        """

        if self.fig is not None:
            plt.close(self.fig)

        # plotting and seeding parameters
        self.fig = None
        self.ax = None
        self.line1 = None
        self.line2 = None
        self.line3 = None
        self.cached_solution = None
        self.graph_previous = None
        self.graph_cost = None

        self.step_counter = 0
        self.n_motion_edges = 0
        self.done = False
        self.last_loc = None
        self.node_history = None

        self._initialize_graph()

        # # initialize robots near targets
        # nearest_landmarks = self.np_random.choice(np.arange(self.n_targets)[self.start_region], size=(self.n_robots,),
        #                                           replace=False)
        # # nearest_landmarks = self.np_random.choice(2 * self.n_robots, size=(self.n_robots,), replace=False)
        # self.x[:self.n_robots, 0:2] = self.x[nearest_landmarks + self.n_robots, 0:2]
        #
        unvisited_targets = np.arange(self.n_targets)[self.unvisited_region] + self.n_robots
        frac_active = np.random.uniform(low=MIN_FRAC_ACTIVE, high=self.frac_active_targets)
        random_unvisited_targets = self.np_random.choice(unvisited_targets,
                                                         size=(int(len(unvisited_targets) * frac_active),),
                                                         replace=False)

        self.visited.fill(1)
        self.visited[random_unvisited_targets] = 0

        self.cached_solution = None
        self.step_counter = 0
        self.done = False
        self.node_history = np.zeros((self.n_agents, 1))
        obs, _, _ = self._get_obs_reward()
        return obs

    @property
    def closest_targets(self):
        r = np.linalg.norm(self.x[:self.n_robots, 0:2].reshape((self.n_robots, 1, 2))
                           - self.x[self.n_robots:, 0:2].reshape((1, self.n_targets, 2)), axis=2)
        closest_targets = np.argmin(r, axis=1) + self.n_robots
        return closest_targets

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space. The robots are in green, the targets in red.
        When a target has been visited, it becomes a blue dot. The plot objects are created on the first render() call
        and persist between calls of this function.
        :param mode: 'human' mode renders the environment, and nothing happens otherwise
        """
        if mode is not 'human':
            return

        if self.fig is None:
            # initialize plot parameters
            plt.ion()
            fig = plt.figure()
            self.ax = fig.add_subplot(111)

            for (i, j) in zip(self.motion_edges[0], self.motion_edges[1]):
                self.ax.plot([self.x[i, 0], self.x[j, 0]], [self.x[i, 1], self.x[j, 1]], 'b')

            # plot robots and targets and visited targets as scatter plot
            line2, = self.ax.plot(self.x[self.n_robots:, 0], self.x[self.n_robots:, 1], 'ro', markersize=12)
            line3, = self.ax.plot([], [], 'b.')
            line1, = self.ax.plot(self.x[0:self.n_robots, 0], self.x[0:self.n_robots, 1], 'go', markersize=20,
                                  linewidth=0)

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

    def _initialize_graph(self):
        """
        Initialization code that is needed after params are re-loaded
        """
        lattice = generate_lattice((self.x_min, self.x_max, self.y_min, self.y_max), self.lattice_vectors)
        # targets = reject_collisions(targets, self.obstacles)

        n_cities = 9
        # intercity_radius = self.x_max/6
        roads = generate_geometric_roads(n_cities, self.x_max, self.motion_radius)
        flag = np.min(np.linalg.norm(_get_pos_diff(lattice, roads), axis=2), axis=1) <= (self.motion_radius/1.4)
        targets = lattice[flag, :]

        r = np.linalg.norm(_get_pos_diff(targets), axis=2)
        r[r > self.motion_radius] = 0
        _, labels = connected_components(csgraph=csr_matrix(r), directed=False, return_labels=True)
        targets = targets[labels == np.argmax(np.bincount(labels)), :]

        # targets += np.random.uniform(low=-0.2, high=0.2, size=np.shape(targets))

        self.n_targets = np.shape(targets)[0]
        self.n_agents = self.n_targets + self.n_robots
        self.x = np.zeros((self.n_agents, self.nx))
        self.x[self.n_robots:, 0:2] = targets

        # self.max_edges = self.n_agents * MAX_EDGES
        if PAD_NODES:
            self.max_nodes = MAX_NODES
        else:
            self.max_nodes = self.n_agents

        self.max_edges = self.max_nodes * MAX_EDGES
        self.agent_type = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))

        self.robot_flag = np.vstack((np.ones((self.n_robots, 1)), np.zeros((self.n_targets, 1))))

        self.landmark_flag = np.vstack((np.zeros((self.n_robots, 1)), np.ones((self.n_targets, 1))))
        self.n_actions = N_ACTIONS

        self.edges = np.zeros((self.max_edges, N_EDGE_FEAT), dtype=np.float32)
        self.nodes = np.zeros((self.max_nodes, N_NODE_FEAT), dtype=np.float32)
        self.senders = -1 * np.ones((self.max_edges,), dtype=np.int32)
        self.receivers = -1 * np.ones((self.max_edges,), dtype=np.int32)

        self.node_history = np.zeros((self.n_agents, 1))

        # communication radius squared
        self.comm_radius2 = self.comm_radius * self.comm_radius

        # initialize state matrices
        self.visited = np.ones((self.n_agents, 1))

        self.unvisited_region = [in_obstacle(self.unvisited_ranges, self.x[i, 0], self.x[i, 1]) for i in
                                 range(self.n_robots, self.n_agents)]
        # self.start_region = [in_obstacle(self.start_ranges, self.x[i, 0], self.x[i, 1]) for i in
        #                      range(self.n_robots, self.n_agents)]

        self.start_region = [0 < i <= self.n_robots + 25 for i in range(self.n_robots, self.n_agents)]

        self.agent_ids = np.reshape((range(self.n_agents)), (-1, 1))
        self.agent_ids = np.reshape((range(self.n_agents)), (-1, 1))

        self.motion_edges, self.motion_dist = _get_graph_edges(self.motion_radius, self.x[self.n_robots:, 0:2],
                                                               self_loops=True)
        # cache motion edges
        self.motion_edges = (self.motion_edges[0] + self.n_robots, self.motion_edges[1] + self.n_robots)
        self.n_motion_edges = len(self.motion_edges[0])

        self.senders[:self.n_motion_edges] = self.motion_edges[0]
        self.receivers[:self.n_motion_edges] = self.motion_edges[1]
        self.edges[:self.n_motion_edges, 0] = self.motion_dist.reshape((-1,))

        # problem's observation and action spaces
        self.action_space = spaces.MultiDiscrete([self.n_actions] * self.n_robots)

        if PAD_NODES:
            nodes_space = Box(shape=(self.max_nodes, N_NODE_FEAT), low=-np.Inf, high=np.Inf, dtype=np.float32)
        else:
            nodes_space = Box(shape=(self.n_agents, N_NODE_FEAT), low=-np.Inf, high=np.Inf, dtype=np.float32)

        self.observation_space = gym.spaces.Dict(
            [
                ("nodes", nodes_space),
                ("edges", Box(shape=(self.max_edges, N_EDGE_FEAT), low=-np.Inf, high=np.Inf, dtype=np.float32)),
                ("senders", Box(shape=(self.max_edges, 1), low=0, high=self.n_agents, dtype=np.float32)),
                ("receivers", Box(shape=(self.max_edges, 1), low=0, high=self.n_agents, dtype=np.float32)),
                ("step", Box(shape=(1, 1), low=0, high=EPISODE_LENGTH, dtype=np.float32)),
            ]
        )


    def construct_time_matrix(self, edge_time=1.0):
        """
        Compute the distance between all pairs of nodes in the graph
        :param edges: list of edges provided as (sender, receiver)
        :param edge_time: uniform edge cost, assumed to be 1.0
        :return:
        """
        edges = (self.motion_edges[0] - self.n_robots, self.motion_edges[1] - self.n_robots)
        time_matrix = np.ones((self.n_targets, self.n_targets)) * np.Inf
        prev = np.zeros((self.n_targets, self.n_targets), dtype=int)
        np.fill_diagonal(time_matrix, 0.0)
        np.fill_diagonal(prev, np.array(range(self.n_targets)))

        changed_last_iter = True  # prevents looping forever in disconnected graphs
        while changed_last_iter and np.sum(time_matrix) == np.Inf:
            changed_last_iter = False
            for (sender, receiver) in zip(edges[0], edges[1]):
                new_cost = np.minimum(time_matrix[:, sender] + edge_time, time_matrix[:, receiver])

                prev[:, receiver] = np.where(time_matrix[:, sender] + edge_time < time_matrix[:, receiver],
                                             sender, prev[:, receiver])

                changed_last_iter = changed_last_iter or (not np.array_equal(new_cost, time_matrix[:, receiver]))
                time_matrix[:, receiver] = new_cost

        return time_matrix, prev

    @staticmethod
    def unpack_obs(obs, ob_space):
        assert tf is not None, "Function unpack_obs() is not available if Tensorflow is not imported."

        # assume flattened box
        if PAD_NODES:
            n_nodes = MAX_NODES
        else:
            n_nodes = (ob_space.shape[0] - N_GLOB_FEAT) // (MAX_EDGES * (2 + N_EDGE_FEAT) + N_NODE_FEAT)
        max_edges = MAX_EDGES
        max_n_edges = n_nodes * max_edges
        dim_edges = N_EDGE_FEAT
        dim_nodes = N_NODE_FEAT

        # unpack node and edge data from flattened array
        shapes = ((n_nodes, dim_nodes), (max_n_edges, dim_edges), (max_n_edges, 1), (max_n_edges, 1), (1, N_GLOB_FEAT))
        sizes = [np.prod(s) for s in shapes]
        tensors = tf.split(obs, sizes, axis=1)
        tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
        nodes, edges, senders, receivers, globs = tensors
        batch_size = tf.shape(nodes)[0]
        nodes = tf.reshape(nodes, (-1, dim_nodes))

        # if PAD_NODES:
        #     # compute node mask
        #     # check if the first column of node features != 1
        #     node_mask = tf.not_equal(tf.slice(nodes, [0, 0], size=[1, -1]), -1)
        #     nodes = tf.boolean_mask(nodes, node_mask, axis=0)
        #     nodes = tf.reshape(nodes, (-1, dim_nodes))
        #     n_node = tf.reduce_sum(tf.reshape(tf.cast(node_mask, tf.float32), (batch_size, -1)), axis=1)
        # else:
        n_node = tf.fill((batch_size,), n_nodes)  # assume n nodes is fixed

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

        globs = tf.reshape(globs, (batch_size, N_GLOB_FEAT))

        # cast all indices to int
        n_node = tf.cast(n_node, tf.int32)
        n_edge = tf.cast(n_edge, tf.int32)
        senders = tf.cast(senders, tf.int32)
        receivers = tf.cast(receivers, tf.int32)

        return batch_size, n_node, nodes, n_edge, edges, senders, receivers, globs

    def controller(self, random=False, greedy=GREEDY_CONTROLLER):
        """
        Greedy controller picks the nearest unvisited target
        :return: control action for each robot (global index of agent chosen)
        """
        if random:
            return self.np_random.choice(self.n_actions, size=(self.n_robots, 1))

        # compute greedy solution
        r = np.linalg.norm(self.x[:self.n_robots, 0:2].reshape((self.n_robots, 1, 2))
                           - self.x[self.n_robots:, 0:2].reshape((1, self.n_targets, 2)), axis=2)
        r[:, np.where(self.visited[self.n_robots:] == 1)] = np.Inf
        greedy_loc = np.argmin(r, axis=1) + self.n_robots
        curr_loc = self.closest_targets

        if self.graph_previous is None:
            self.graph_cost, self.graph_previous = self.construct_time_matrix()

        if greedy:
            next_loc = greedy_loc
        else:
            assert ortools is not None, "Vehicle routing controller is not available if OR-Tools is not imported."
            if self.cached_solution is None:
                self.cached_solution = solve_vrp(self)

            next_loc = np.zeros((self.n_robots,), dtype=int)

            for i in range(self.n_robots):

                if len(self.cached_solution[i]) > 1:
                    if curr_loc[i] == self.cached_solution[i][0]:
                        self.cached_solution[i] = self.cached_solution[i][1:]

                    next_loc[i] = self.cached_solution[i][0]
                elif len(self.cached_solution[i]) == 1:
                    if curr_loc[i] == self.cached_solution[i][0]:
                        self.cached_solution[i] = []
                    next_loc[i] = greedy_loc[i]
                else:
                    next_loc[i] = greedy_loc[i]

        # use the precomputed predecessor matrix to select the next node - necessary for avoiding obstacles
        next_loc = self.graph_previous[next_loc - self.n_robots, curr_loc - self.n_robots] + self.n_robots
        u_ind = np.zeros((self.n_robots, 1), dtype=np.int32)
        for i in range(self.n_robots):
            u_ind[i] = np.where(self.mov_edges[1][np.where(self.mov_edges[0] == i)] == next_loc[i])[0]

        return u_ind
