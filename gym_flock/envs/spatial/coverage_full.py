from gym_flock.envs.spatial.coverage_arl import CoverageARLEnv

DOWNSAMPLE_RATE = 10
# PERIMETER_DELTA = 12.0
PERIMETER_DELTA = 2.0


class CoverageFullEnv(CoverageARLEnv):

    def __init__(self):
        """Initialize the mapping environment
        """

        super(CoverageFullEnv, self).__init__(n_robots=10, episode_length=10000, pad_nodes=False, max_nodes=1500,
                                              nearby_starts=True, num_subgraphs=1, check_connected=True,
                                              downsample_rate=DOWNSAMPLE_RATE, perimeter_delta=PERIMETER_DELTA,
                                              horizon=19)
