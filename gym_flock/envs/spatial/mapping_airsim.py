import airsim
import numpy as np
from gym_flock.envs.airsim.utils import send_loc_commands, send_velocity_commands, setup_drones, get_states
from gym_flock.envs.spatial.make_map import gen_obstacle_grid
from gym_flock.envs.airsim.utils import parse_settings
from gym_flock.envs.spatial.mapping_rad import MappingRadEnv
from gym_flock.envs.spatial.utils import _get_pos_diff

# parameters for map generation
ranges = [(5, 30),  (35, 65), (70, 95)]
OBST = gen_obstacle_grid(ranges)

XMAX = 100
YMAX = 100

unvisited_regions = [(0, 35, 30, 100), (65, 100, 0, 35)]
start_regions = [(0, 35, 0, 35)]


class MappingAirsimEnv(MappingRadEnv):

    def __init__(self):
        # parse settings file with drone names and home locations
        fname = '/home/kate/Documents/AirSim/settings.json'
        self.names, self.home = parse_settings(fname)

        super(MappingAirsimEnv, self).__init__(n_robots=len(self.names), obstacles=OBST, xmax=XMAX, ymax=YMAX)

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()

        self.z = np.linspace(start=-50, stop=-30, num=len(self.names))
        self.episode_length = 100000
        self.a_max = 10.0

    def reset(self):
        self.client.reset()
        setup_drones(self.client, self.names)

        self.x[:self.n_robots, 2:4] = self.np_random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_robots, 2))

        # initialize robots near targets
        nearest_landmarks = self.np_random.choice(np.arange(self.n_targets)[self.start_region], size=(self.n_robots,), replace=False)
        self.x[:self.n_robots, 0:2] = self.x[nearest_landmarks + self.n_robots, 0:2]
        self.x[:self.n_robots, 0:2] += self.np_random.uniform(low=-0.5 * self.motion_radius,
                                                              high=0.5 * self.motion_radius, size=(self.n_robots, 2))

        self.visited.fill(1)
        self.visited[np.arange(self.n_targets)[self.unvisited_region] + self.n_robots] = 0

        send_loc_commands(self.client, self.names, self.home, self.x[:self.n_robots, 0:2], self.z)

        states, _ = get_states(self.client, self.names, self.home)
        self.x[:self.n_robots, :] = states
        self.cached_solution = None
        self.step_counter = 0
        obs, _, _ = self._get_obs_reward()

        return obs

    def step(self, u_ind):
        # action will be the index of the neighbor in the graph
        u_ind = np.reshape(u_ind, (-1, 1))
        robots_index = np.reshape(range(self.n_robots), (-1, 1))
        u_ind = np.reshape(self.mov_edges[1], (self.n_robots, self.n_actions))[robots_index, u_ind]
        # new_waypoint = np.reshape(self.x[u_ind, 0:2], (self.n_robots, 2))

        diff = _get_pos_diff(self.x[:self.n_robots, 0:2], self.x[:, 0:2])
        u = -1.0 * diff[robots_index, u_ind, 0:2].reshape((self.n_robots, 2))

        u = self.action_gain * np.clip(u, a_min=-self.a_max, a_max=self.a_max)
        u = np.reshape(u, (self.n_robots, 2))

        # send_loc_commands(self.client, self.names, self.home, new_waypoint, self.z)
        send_velocity_commands(self.client, self.names, self.z, u, duration=0.2)

        states, _ = get_states(self.client, self.names, self.home)
        self.x[:self.n_robots, :] = states  # get drone locations and velocities
        obs, reward, done = self._get_obs_reward()
        return obs, reward, done, {}

