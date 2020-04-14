from airsim.client import MultirotorClient
from gym_flock.envs.airsim.utils import send_loc_commands, send_velocity_commands, setup_drones, get_states
from gym_flock.envs.airsim.utils import parse_settings

import numpy as np
import copy

from gym_flock.envs.spatial.make_map import gen_obstacle_grid
from gym_flock.envs.spatial.mapping_rad import MappingRadEnv
# from gym_flock.envs.spatial.utils import _get_pos_diff

# parameters for map generation
ranges = [(5, 30), (35, 65), (70, 95)]
OBST = gen_obstacle_grid(ranges)

XMAX = 100
YMAX = 100

FRAC_ACTIVE = 0.5
MIN_FRAC_ACTIVE = 0.5

# unvisited_regions = [(0, 35, 30, 100), (65, 100, 0, 35)]
unvisited_regions = [(0, 100, 0, 100)]
start_regions = [(0, 100, 0, 100)]


class MappingAirsimEnv(MappingRadEnv):

    def __init__(self):
        # parse settings file with drone names and home locations
        fname = '/home/kate/Documents/AirSim/settings.json'
        self.names, self.home = parse_settings(fname)

        super(MappingAirsimEnv, self).__init__(n_robots=len(self.names), obstacles=OBST, xmax=XMAX, ymax=YMAX,
                                               starts=start_regions, unvisiteds=unvisited_regions)

        # connect to the AirSim simulator
        self.client = MultirotorClient()
        self.client.confirmConnection()

        self.actual_x = np.zeros((self.n_robots, 2))

        self.z = np.linspace(start=-50, stop=-30, num=len(self.names))
        self.episode_length = 100000
        self.v_max = 2.0

    def reset(self):
        print('Re-setting drones...')
        self.client.reset()
        setup_drones(self.client, self.names)
        print('Drones are set up...')

        self.last_loc = None

        # initialize robots near targets
        nearest_landmarks = self.np_random.choice(np.arange(self.n_targets)[self.start_region], size=(self.n_robots,),
                                                  replace=False)
        self.x[:self.n_robots, 0:2] = self.x[nearest_landmarks + self.n_robots, 0:2]

        unvisited_targets = np.arange(self.n_targets)[self.unvisited_region] + self.n_robots
        frac_active = np.random.uniform(low=MIN_FRAC_ACTIVE, high=self.frac_active_targets)
        random_unvisited_targets = self.np_random.choice(unvisited_targets,
                                                         size=(int(len(unvisited_targets) * frac_active),),
                                                         replace=False)

        self.visited.fill(1)
        self.visited[random_unvisited_targets] = 0

        print('Moving drones to initial positions...')
        send_loc_commands(self.client, self.names, self.home, self.x[:self.n_robots, 0:2], self.z)
        print('Drones are in position...')
        self._update_states()
        self.cached_solution = None
        self.step_counter = 0
        self.done = False
        self.node_history = np.zeros((self.n_agents, 1))

        obs, _, _ = self._get_obs_reward()

        return obs

    def _update_states(self):
        states, _ = get_states(self.client, self.names, self.home)
        self.x[:self.n_robots, :] = states[:, 0:2]
        self.actual_x[:, 0:2] = states[:, 0:2]
        self.x[:self.n_robots, 0:2] = self.x[self.closest_targets, 0:2]

    def step(self, u_ind):

        old_last_loc = self.last_loc
        self.last_loc = self.closest_targets

        # action will be the index of the neighbor in the graph
        next_loc = copy.copy(u_ind.reshape((-1, 1)))
        for i in range(self.n_robots):
            next_loc[i] = self.mov_edges[1][np.where(self.mov_edges[0] == i)][u_ind[i]]

        self._update_states()

        # proportional controller converts position offset to velocity commands
        u = self.actual_x - np.reshape(self.x[next_loc, 0:2], (self.n_robots, 2))
        u = -1.0 * np.clip(u, a_min=-self.v_max, a_max=self.v_max)
        send_velocity_commands(self.client, self.names, self.z, u, duration=0.1)

        # next_waypoint = np.reshape(self.x[next_loc, 0:2], (self.n_robots, 2))
        # send_loc_commands(self.client, self.names, self.home, next_waypoint, self.z)

        self._update_states()

        # if stayed in the same spot, don't update last loc
        self.last_loc = np.where(self.last_loc == self.closest_targets, old_last_loc, self.last_loc)

        obs, reward, done = self._get_obs_reward()
        return obs, reward, done, {}
