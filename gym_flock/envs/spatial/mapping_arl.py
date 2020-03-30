from gym_flock.envs.spatial.mapping_rad import MappingRadEnv
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from gym.spaces import Box

from gym_flock.envs.spatial.utils import _get_graph_edges, _get_pos_diff

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}

# number of node and edge features
N_NODE_FEAT = 4
N_EDGE_FEAT = 2
N_GLOB_FEAT = 1

COMM_EDGES = False

# padding for a variable number of graph edges
PAD_NODES = True
MAX_NODES = 1500
# MAX_NODES = 1300
MAX_EDGES = 3

# number of edges/actions for each robot, fixed
N_ACTIONS = 4

EPISODE_LENGTH = 100000
EARLY_TERMINATION = False

N_ROBOTS = 10

FRAC_ACTIVE = 0.5

DELTA = 5.5


class MappingARLEnv(MappingRadEnv):

    def __init__(self, n_robots=N_ROBOTS, frac_active_targets=FRAC_ACTIVE):
        """Initialize the mapping environment
        """
        super(MappingRadEnv, self).__init__()

        self.episode_length = EPISODE_LENGTH

        self.res = 5

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

    def update_state(self, state):
        self.x[:self.n_robots, :] = state
        self.x[:self.n_robots, 0:2] = self.x[self.closest_targets, 0:2]

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

        # initialize robots near targets
        # nearest_landmarks = self.np_random.choice(np.arange(self.n_targets)[self.start_region], size=(self.n_robots,),
        #                                           replace=False)
        nearest_landmarks = self.np_random.choice(self.n_targets, size=(self.n_robots,), replace=False)
        self.x[:self.n_robots, 0:2] = self.x[nearest_landmarks + self.n_robots, 0:2]

        unvisited_targets = np.arange(self.n_targets)[self.unvisited_region] + self.n_robots

        random_unvisited_targets = self.np_random.choice(unvisited_targets,
                                                         size=(int(len(unvisited_targets) * self.frac_active_targets),),
                                                         replace=False)

        self.visited.fill(1)
        self.visited[random_unvisited_targets] = 0

        self.cached_solution = None
        self.step_counter = 0
        self.done = False
        self.node_history = np.zeros((self.n_agents, 1))
        obs, _, _ = self._get_obs_reward()
        return obs

    def step(self, action):
        self.last_loc = self.closest_targets

        next_loc = copy.copy(action)
        for i in range(self.n_robots):
            next_loc[i] = self.mov_edges[1][np.where(self.mov_edges[0] == i)][action[i]]

        self.x[:self.n_robots, 0:2] = self.x[next_loc.flatten(), 0:2]
        obs, reward, done = self._get_obs_reward()
        return obs, reward, done, {}

    def _initialize_graph(self):
        """
        Initialization code that is needed after params are re-loaded
        """

        targets = from_occupancy()

        # keep the largest connected sub-graph
        r = np.linalg.norm(_get_pos_diff(targets), axis=2)
        r[r > self.motion_radius] = 0
        _, labels = connected_components(csgraph=csr_matrix(r), directed=False, return_labels=True)
        targets = targets[labels == np.argmax(np.bincount(labels)), :]

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

        self.unvisited_region = [True] * (self.n_agents - self.n_robots)

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


def from_occupancy():
    fname = '/home/kate/work/gym-flock/grid_slice.npy'

    arr = np.load(fname)

    xs = np.array(range(arr.shape[0]))
    ys = np.array(range(arr.shape[1]))

    xs, ys = np.meshgrid(xs, ys)

    xs = xs.flatten()
    ys = ys.flatten()

    not_occupied = [not arr[i, j] for i, j in zip(xs, ys)]
    occupied = [not i for i in not_occupied]

    xs_nocc = np.reshape(xs[not_occupied], (-1, 1))
    ys_nocc = np.reshape(ys[not_occupied], (-1, 1))
    vertices = np.hstack((xs_nocc, ys_nocc))

    xs_occ = np.reshape(xs[occupied], (-1, 1))
    ys_occ = np.reshape(ys[occupied], (-1, 1))
    vertices_occ = np.hstack((xs_occ, ys_occ))

    flag = np.min(np.linalg.norm(_get_pos_diff(vertices, vertices_occ), axis=2), axis=1) <= 2.0

    targets = vertices[flag, :]

    downsample_rate = 10
    # xyz_min = [-321.0539855957031, -276.5395050048828, -9.511598587036133]
    xyz_min = np.reshape(np.array([-321.0539855957031, -276.5395050048828]), (1, 2))
    # xyz_max = [319.4460144042969, 277.9604949951172, 23.488401412963867]
    res = np.reshape(np.array([0.5, 0.5]), (1, 2)) * downsample_rate  # [0.5, 0.5, 1.0]
    targets = targets * res + xyz_min + res/2

    # nearest_landmarks = np.random.choice(np.arange(np.shape(targets)[0]), size=(5,), replace=False)
    # nearest_landmarks = self.np_random.choice(2 * self.n_robots, size=(self.n_robots,), replace=False)
    # print(targets[nearest_landmarks + 5, 0:2])


    return targets
