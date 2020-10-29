from gym_flock.envs.spatial.coverage_arl import CoverageARLEnv

DOWNSAMPLE_RATE = 10
PERIMETER_DELTA = 12.0


class ExploreFullEnv(CoverageARLEnv):

    def __init__(self):
        """Initialize the mapping environment
        """

        super(ExploreFullEnv, self).__init__(hide_nodes=True, n_node_feat=4, n_robots=100, episode_length=50,
                                             pad_nodes=False, max_nodes=1500,
                                             nearby_starts=True, num_subgraphs=1, check_connected=True,
                                             downsample_rate=DOWNSAMPLE_RATE, perimeter_delta=PERIMETER_DELTA,
                                             horizon=19)
