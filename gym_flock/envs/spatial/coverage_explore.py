from gym_flock.envs.spatial.coverage_arl import CoverageARLEnv


class ExploreEnv(CoverageARLEnv):

    def __init__(self):
        """Initialize the mapping environment
        """

        super(ExploreEnv, self).__init__(hide_nodes=True, n_node_feat=4, horizon=19, episode_length=50)
