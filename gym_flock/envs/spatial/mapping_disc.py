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

from gym_flock.envs.spatial.make_map import generate_lattice, reject_collisions, gen_obstacle_grid, in_obstacle
from gym_flock.envs.spatial.vrp_solver import solve_vrp
from gym_flock.envs.spatial.utils import _get_pos_diff, _get_graph_edges, _get_k_edges

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
N_NODE_FEAT = 2
N_EDGE_FEAT = 1
N_GLOB_FEAT = 1

# padding for a variable number of graph edges
PAD_NODES = False
MAX_NODES = 300
MAX_EDGES = 3

# number of edges/actions for each robot, fixed
N_ACTIONS = 4
ALLOW_NEAREST = False
GREEDY_CONTROLLER = False
# GREEDY_CONTROLLER = True

EPISODE_LENGTH = 30

# parameters for map generation
# ranges = [(5, 30),  (35, 65), (70, 95)]
ranges = [(5, 25), (30, 50), (57, 75), (80, 95)]
# ranges = [(5, 65), (70, 130), (135, 195)]
# ranges = [(5, 50), (55, 100), (110, 150), (160, 195)]
# ranges = [(5, 50), (55, 100), (105, 150), (155, 195),(205, 250), (260, 300), (310, 355), (360, 395)]

OBST = gen_obstacle_grid(ranges)

N_ROBOTS = 3
XMAX = 100
YMAX = 100
# XMAX = 200
# YMAX = 200
# XMAX = 400
# YMAX = 400

# FRAC_ACTIVE = 1.0
FRAC_ACTIVE = 0.75

# unvisited_regions = [(0, 200, 200, 400), (200, 400, 0, 200)]
# start_regions = [(75, 125, 150, 200)]

# unvisited_regions = [(0, 70, 60, 200), (130, 200, 0, 200)]
# start_regions = [(0, 70, 0, 70)]
#
unvisited_regions = [(0, 30, 25, 100), (55, 100, 0, 57)]
# unvisited_regions = [(0, 35, 30, 70), (65, 100, 0, 70)]

# start_regions = [(30, 70, 30, 70)]
start_regions = [(0, 25, 0, 25)]


class MappingDiscEnv(gym.Env):
    def __init__(self, n_robots=N_ROBOTS, frac_active_targets=FRAC_ACTIVE, obstacles=OBST, xmax=XMAX, ymax=YMAX):
        """Initialize the mapping environment
        """
        super(MappingDiscEnv, self).__init__()

        self.y_min = 0
        self.x_min = 0
        self.x_max = xmax
        self.y_max = ymax
        self.obstacles = obstacles

        # # triangular lattice
        # self.lattice_vectors = [
        #     2.75 * np.array([-1.414, -1.414]),
        #     2.75 * np.array([-1.414, 1.414])]

        # square lattice
        self.lattice_vectors = [
            np.array([-5.5, 0.]),
            np.array([0., -5.5])]

        self.np_random = None
        self.seed()

        # dim of state per agent, 2D position and 2D velocity
        self.nx = 4
        self.velocity_control = True

        # agent dynamics are controlled with 2D acceleration
        self.nu = 2

        # number of robots and targets
        self.n_robots = n_robots
        self.frac_active_targets = frac_active_targets

        # dynamics parameters
        # self.dt = 1.0
        self.dt = 2.0
        self.n_steps = 5
        self.ddt = self.dt / self.n_steps
        self.v_max = 5.0  # max velocity
        self.a_max = 5.0  # max acceleration
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
        self.sensor_radius = 5.0  # 2.0

        # call helper function to initialize arrays
        # self.system_changed = True
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

        self.episode_length = EPISODE_LENGTH
        self.step_counter = 0
        self.n_motion_edges = 0

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

        for _ in range(self.n_steps):
            diff = _get_pos_diff(self.x[:self.n_robots, 0:2], self.x[:, 0:2])
            u = -1.0 * diff[robots_index, u_ind, 0:2].reshape((self.n_robots, 2))

            if self.velocity_control:
                u = self.action_gain * np.clip(u, a_min=-self.v_max, a_max=self.v_max)
                self.x[:self.n_robots, 0:2] = self.x[:self.n_robots, 0:2] + u[:, 0:2] * self.ddt
            else:
                u = self.action_gain * np.clip(u, a_min=-self.a_max, a_max=self.a_max)
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

        # action edges from landmarks to robots
        action_edges, action_dist = _get_k_edges(self.n_actions, self.x[:self.n_robots, 0:2],
                                                 self.x[self.n_robots:, 0:2], allow_nearest=ALLOW_NEAREST)
        action_edges = (action_edges[0], action_edges[1] + self.n_robots)
        assert len(action_edges[0]) == N_ACTIONS * self.n_robots, "Number of action edges is not num robots x n_actions"
        self.mov_edges = action_edges

        # planning edges from robots to landmarks
        plan_edges, plan_dist = _get_graph_edges(self.motion_radius, self.x[:self.n_robots, 0:2],
                                                 self.x[self.n_robots:, 0:2])
        plan_edges = (plan_edges[0], plan_edges[1] + self.n_robots)

        # communication edges among robots
        comm_edges, comm_dist = _get_graph_edges(self.comm_radius, self.x[:self.n_robots, 0:2])

        # which landmarks is the robot observing?
        sensor_edges, _ = _get_graph_edges(self.sensor_radius, self.x[:self.n_robots, 0:2],
                                           self.x[self.n_robots:, 0:2])
        old_sum = np.sum(self.visited[self.n_robots:])
        self.visited[sensor_edges[1] + self.n_robots] = 1

        # we want to fix the number of edges into the robot from targets.
        # senders = np.concatenate((plan_edges[0], action_edges[1], comm_edges[0], self.motion_edges[0]))
        # receivers = np.concatenate((plan_edges[1], action_edges[0], comm_edges[1], self.motion_edges[1]))
        # edges = np.concatenate((plan_dist, action_dist, comm_dist, self.motion_dist)).reshape((-1, N_EDGE_FEAT))
        senders = np.concatenate((plan_edges[0], action_edges[1], comm_edges[0]))
        receivers = np.concatenate((plan_edges[1], action_edges[0], comm_edges[1]))
        edges = np.concatenate((plan_dist, action_dist, comm_dist)).reshape((-1, N_EDGE_FEAT))
        assert len(senders) + self.n_motion_edges <= np.shape(self.senders)[0], "Increase MAX_EDGES"

        # -1 indicates unused edges
        self.senders[self.n_motion_edges:] = -1
        self.receivers[self.n_motion_edges:] = -1
        self.nodes.fill(-1)

        # self.senders[self.n_motion_edges:self.n_motion_edges + len(senders)] = senders
        # self.receivers[self.n_motion_edges:self.n_motion_edges + len(receivers)] = receivers
        # self.edges[self.n_motion_edges:self.n_motion_edges + len(senders), :] = edges

        self.senders[-len(senders):] = senders
        self.receivers[-len(receivers):] = receivers
        self.edges[-len(senders):, :] = edges

        self.nodes[0:self.n_agents, 0] = self.agent_type.flatten()
        self.nodes[0:self.n_agents, 1] = np.logical_not(self.visited).flatten()
        # TODO landmark data will grow from beginning to end, while the robot data goes at the end

        step_array = np.array([self.step_counter]).reshape((1, 1))

        obs = {'nodes': self.nodes, 'edges': self.edges, 'senders': self.senders, 'receivers': self.receivers,
               'step': step_array}

        self.step_counter += 1
        done = self.step_counter == self.episode_length or np.sum(self.visited[self.n_robots:]) == self.n_targets
        # reward = np.sum(self.visited[self.n_robots:]) - self.n_targets if done else 0.
        # reward = (np.sum(self.visited[self.n_robots:]) - self.n_targets) / self.n_targets
        reward = np.sum(self.visited[self.n_robots:]) - old_sum
        return obs, reward, done

    def reset(self):
        """
        Reset system state. Agents are initialized in a square with side self.r_max
        :return: observations, adjacency matrix
        """
        self.x[:self.n_robots, 2:4] = self.np_random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots, 2))

        # initialize robots near targets
        nearest_landmarks = self.np_random.choice(np.arange(self.n_targets)[self.start_region], size=(self.n_robots,),
                                                  replace=False)
        # nearest_landmarks = self.np_random.choice(2 * self.n_robots, size=(self.n_robots,), replace=False)
        self.x[:self.n_robots, 0:2] = self.x[nearest_landmarks + self.n_robots, 0:2]
        self.x[:self.n_robots, 0:2] += self.np_random.uniform(low=-0.5 * self.motion_radius,
                                                              high=0.5 * self.motion_radius, size=(self.n_robots, 2))

        # self.visited.fill(1)
        # self.visited[self.np_random.choice(self.n_targets, size=(int(self.n_targets * self.frac_active_targets),),
        #                                    replace=False) + self.n_robots] = 0

        self.visited.fill(1)
        self.visited[np.arange(self.n_targets)[self.unvisited_region] + self.n_robots] = 0

        self.cached_solution = None
        self.step_counter = 0
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
        targets = generate_lattice((self.x_min, self.x_max, self.y_min, self.y_max), self.lattice_vectors)
        targets = reject_collisions(targets, self.obstacles)

        targets += np.random.uniform(low=-0.2, high=0.2, size=np.shape(targets))

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
        self.n_actions = N_ACTIONS

        self.edges = np.zeros((self.max_edges, 1), dtype=np.float32)
        self.nodes = np.zeros((self.max_nodes, 2), dtype=np.float32)
        self.senders = -1 * np.ones((self.max_edges,), dtype=np.int32)
        self.receivers = -1 * np.ones((self.max_edges,), dtype=np.int32)

        # communication radius squared
        self.comm_radius2 = self.comm_radius * self.comm_radius

        # initialize state matrices
        self.visited = np.ones((self.n_agents, 1))

        self.unvisited_region = [in_obstacle(unvisited_regions, self.x[i, 0], self.x[i, 1]) for i in
                                 range(self.n_robots, self.n_agents)]
        self.start_region = [in_obstacle(start_regions, self.x[i, 0], self.x[i, 1]) for i in
                             range(self.n_robots, self.n_agents)]

        self.agent_ids = np.reshape((range(self.n_agents)), (-1, 1))

        # caching distance computation
        self.diff = np.zeros((self.n_agents, self.n_agents, self.nx))
        self.r2 = np.zeros((self.n_agents, self.n_agents))

        self.motion_edges, self.motion_dist = _get_graph_edges(self.motion_radius, self.x[self.n_robots:, 0:2],
                                                               self_loops=True)
        # cache motion edges
        self.motion_edges = (self.motion_edges[0] + self.n_robots, self.motion_edges[1] + self.n_robots)
        self.n_motion_edges = len(self.motion_edges[0])

        self.senders[:self.n_motion_edges] = self.motion_edges[0]
        self.receivers[:self.n_motion_edges] = self.motion_edges[1]
        self.edges[:self.n_motion_edges, :] = self.motion_dist.reshape((-1, 1))

        # problem's observation and action spaces
        if self.n_robots == 1:
            self.action_space = spaces.Discrete(self.n_actions)
        else:
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
        dim_edges = 1
        dim_nodes = 2

        # unpack node and edge data from flattened array
        shapes = ((n_nodes, dim_nodes), (max_n_edges, dim_edges), (max_n_edges, 1), (max_n_edges, 1), (1, N_GLOB_FEAT))
        sizes = [np.prod(s) for s in shapes]
        tensors = tf.split(obs, sizes, axis=1)
        tensors = [tf.reshape(t, (-1,) + s) for (t, s) in zip(tensors, shapes)]
        nodes, edges, senders, receivers, globs = tensors
        batch_size = tf.shape(nodes)[0]
        nodes = tf.reshape(nodes, (-1, dim_nodes))

        if PAD_NODES:
            # compute node mask
            node_mask = tf.not_equal(tf.slice(nodes, [0, 0], size=[1, -1]), -1)
            nodes = tf.boolean_mask(nodes, node_mask, axis=0)
            nodes = tf.reshape(nodes, (-1, dim_nodes))
            n_node = tf.reduce_sum(tf.reshape(tf.cast(node_mask, tf.float32), (batch_size, -1)), axis=1)
        else:
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
                if len(self.cached_solution[i]) == 1:  # if out of vrp waypoints, use greedy waypoint
                    next_loc[i] = greedy_loc[i]
                else:  # use vrp solution
                    if curr_loc[i] == self.cached_solution[i][0]:
                        self.cached_solution[i] = self.cached_solution[i][1:]
                    next_loc[i] = self.cached_solution[i][0]

        # use the precomputed predecessor matrix to select the next node - necessary for avoiding obstacles
        next_loc = self.graph_previous[next_loc - self.n_robots, curr_loc - self.n_robots] + self.n_robots

        # now pick the closest immediate neighbor
        # TODO - is this necessary? should be easier to grab the index of next_loc in mov_edges
        r = np.linalg.norm(self.x[next_loc, 0:2].reshape((self.n_robots, 1, 2))
                           - self.x[:, 0:2].reshape((1, self.n_agents, 2)), axis=2)

        closest_neighbor = np.argmin(np.reshape(r[self.mov_edges], (self.n_robots, N_ACTIONS)), axis=1)

        return closest_neighbor
