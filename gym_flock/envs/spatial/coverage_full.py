from gym_flock.envs.spatial.coverage import CoverageEnv
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import copy
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.pyplot import gca
from gym.spaces import Box

from gym_flock.envs.spatial.utils import _get_graph_edges, _get_pos_diff
from gym_flock.envs.spatial.make_map import from_occupancy

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix

EPISODE_LENGTH = 100000

N_ROBOTS = 10

NEARBY_STARTS = False
NUM_SUBGRAPHS = 1
MIN_GRAPH_SIZE = 200
DOWNSAMPLE_RATE = 10
PERIMETER_DELTA = 2.0
CHECK_CONNECTED = True

# Trying to generalize to higher res graph?
# DOWNSAMPLE_RATE = 5
# PERIMETER_DELTA = 4.0
# CHECK_CONNECTED = False

# padding for a variable number of graph edges
PAD_NODES = False
MAX_NODES = 1500

MAP_RES = 0.5


class CoverageFullEnv(CoverageEnv):

    def __init__(self, n_robots=N_ROBOTS):
        """Initialize the mapping environment
        """

        super(CoverageFullEnv, self).__init__(n_robots=n_robots, init_graph=False, episode_length=EPISODE_LENGTH,
                                             res=MAP_RES * DOWNSAMPLE_RATE, pad_nodes=PAD_NODES, max_nodes=MAX_NODES,
                                             nearby_starts=NEARBY_STARTS)

        # need to initialize graph to set up the observation space
        self.all_targets = None
        self.load_graph()
        targets, _ = self._generate_targets()
        self._initialize_graph(targets)

    def update_state(self, state):
        self.x[:self.n_robots, :] = state
        self.x[:self.n_robots, 0:2] = self.x[self.closest_targets, 0:2]

    def load_graph(self):
        targets = from_occupancy(downsample_rate=DOWNSAMPLE_RATE, perimeter_delta=PERIMETER_DELTA)

        if CHECK_CONNECTED:
            # keep the largest connected sub-graph
            r = np.linalg.norm(_get_pos_diff(targets), axis=2)
            r[r > self.motion_radius] = 0
            _, labels = connected_components(csgraph=csr_matrix(r), directed=False, return_labels=True)
            self.all_targets = targets[labels == np.argmax(np.bincount(labels)), :]
        else:
            self.all_targets = targets

    def _generate_targets(self):
        return self.all_targets, False
