import numpy as np
from gym_flock.envs.spatial.coverage import CoverageEnv
from gym_flock.envs.spatial.utils import _get_pos_diff
from gym_flock.envs.spatial.make_map import from_occupancy

from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


MIN_GRAPH_SIZE = 200
MAP_RES = 0.5


class CoverageARLEnv(CoverageEnv):

    def __init__(self, n_robots=3, episode_length=75, pad_nodes=True, max_nodes=1000,
                 nearby_starts=True, num_subgraphs=3, check_connected=True,
                 downsample_rate=10, perimeter_delta=2.0, horizon=15):
        """Initialize the mapping environment
        """

        super(CoverageARLEnv, self).__init__(n_robots=n_robots, init_graph=False, episode_length=episode_length,
                                             res=MAP_RES * downsample_rate, pad_nodes=pad_nodes, max_nodes=max_nodes,
                                             nearby_starts=nearby_starts, horizon=horizon)

        # need to initialize graph to set up the observation space
        self.check_connected = check_connected
        self.downsample_rate = downsample_rate
        self.perimeter_delta = perimeter_delta
        self.num_subgraphs = num_subgraphs
        self.all_targets = None
        self.min_xy = None
        self.max_xy = None
        self.range_xy = None
        self.subgraph_size = None
        self.load_graph()
        targets, _ = self._generate_targets()
        self._initialize_graph(targets)

    def update_state(self, state):
        self.x[:self.n_robots, :] = state
        self.x[:self.n_robots, 0:2] = self.x[self.closest_targets, 0:2]

    def load_graph(self):
        targets = from_occupancy(downsample_rate=self.downsample_rate, perimeter_delta=self.perimeter_delta)

        if self.check_connected:
            # keep the largest connected sub-graph
            r = np.linalg.norm(_get_pos_diff(targets), axis=2)
            r[r > self.motion_radius] = 0
            _, labels = connected_components(csgraph=csr_matrix(r), directed=False, return_labels=True)
            self.all_targets = targets[labels == np.argmax(np.bincount(labels)), :]
        else:
            self.all_targets = targets

        if self.num_subgraphs > 1:
            self.min_xy = np.min(self.all_targets, axis=0).reshape((1, 2))
            self.max_xy = np.max(self.all_targets, axis=0).reshape((1, 2))
            self.range_xy = self.max_xy - self.min_xy
            self.subgraph_size = self.range_xy / self.num_subgraphs

    def _generate_targets(self):
        if self.num_subgraphs > 1:
            n_targets = 0
            targets = None
            while n_targets < MIN_GRAPH_SIZE:
                graph_start = np.random.uniform(low=self.min_xy, high=self.max_xy - self.subgraph_size)
                graph_end = graph_start + self.subgraph_size
                targets = self.all_targets[
                          np.all(np.logical_and(graph_start <= self.all_targets, self.all_targets < graph_end), axis=1),
                          :]
                if np.shape(targets)[0] < MIN_GRAPH_SIZE:
                    continue

                r = np.linalg.norm(_get_pos_diff(targets), axis=2)
                r[r > self.motion_radius] = 0
                _, labels = connected_components(csgraph=csr_matrix(r), directed=False, return_labels=True)
                targets = targets[labels == np.argmax(np.bincount(labels)), :]
                n_targets = np.shape(targets)[0]
            return targets, True
        return self.all_targets, False
