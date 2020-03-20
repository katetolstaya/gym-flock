from gym_flock.envs.spatial.mapping_rad import MappingRadEnv


class MappingARLEnv(MappingRadEnv):

    def __init__(self):
        super(MappingARLEnv, self).__init__(n_robots=5)
        self.episode_length = 100000

    def update_state(self, state):
        self.x[:self.n_robots, :] = state
        self.x[:self.n_robots, 0:2] = self.x[self.closest_targets, 0:2]
