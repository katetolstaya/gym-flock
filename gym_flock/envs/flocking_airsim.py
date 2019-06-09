import airsim
import numpy as np
from time import sleep
import re
from gym_flock.envs.flocking_relative import FlockingRelativeEnv

# N - number of drones
# dist - dist between drones on circumference, 0.5 < 0.75 keeps things interesting
def circle_helper(N, dist):
    r = dist * N / 2 / np.pi
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).reshape((N, 1))
    # angles2 = np.pi - angles
    return r * np.hstack((np.cos(angles), np.sin(angles))), -0.5 * np.hstack((np.cos(angles), -0.5 * np.sin(angles)))


def circle(N):
    if N <= 20:
        return circle_helper(N, 0.5)
    else:
        smalln = int(N * 2.0 / 5.0)
        circle1, v1 = circle_helper(smalln, 0.5)
        circle2, v2 = circle_helper(N - smalln, 0.5)
        return np.vstack((circle1, circle2)), np.vstack((v1, v2))


def grid(N, side=5):
    side2 = int(N / side)
    xs = np.arange(0, side) - side / 2.0
    ys = np.arange(0, side2) - side2 / 2.0
    xs, ys = np.meshgrid(xs, ys)
    xs = xs.reshape((N, 1))
    ys = ys.reshape((N, 1))
    return 0.6 * np.hstack((xs, ys))


def twoflocks(N):
    half_n = int(N / 2)
    grid1 = grid(half_n)
    delta = 6
    grid2 = grid1.copy() + np.array([0, delta / 2]).reshape((1, 2))
    grid1 = grid1 + np.array([0, -delta / 2]).reshape((1, 2))

    vels1 = np.tile(np.array([-1.0, delta]).reshape((1, 2)), (half_n, 1))
    vels2 = np.tile(np.array([1.0, -delta]).reshape((1, 2)), (half_n, 1))

    grids = np.vstack((grid1, grid2))
    velss = 0.05 * np.vstack((vels1, vels2))

    return grids, velss


def parse_settings(fname):
    names = []
    homes = []
    for line in open(fname):
        for n in re.findall(r'\"(.+?)\": {', line):
            if n != 'Vehicles':
                names.append(n)
        p = re.findall(r'"X": ([-+]?\d*\.*\d+), "Y": ([-+]?\d*\.*\d+), "Z": ([-+]?\d*\.*\d+)', line)
        if p:
            homes.append(np.array([float(p[0][0]), float(p[0][1]), float(p[0][2])]).reshape((1, 3)))
    return names, np.concatenate(homes, axis=0)


class FlockingAirsimEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingAirsimEnv, self).__init__()

        # parse settings file with drone names and home locations
        fname = '/home/kate/Documents/AirSim/settings.json'
        self.names, self.home = parse_settings(fname)
        self.n_agents = len(self.names)

        # rescale locations and velocities to avoid changing potential function
        self.scale = 6.0

        # duration of velocity commands
        self.true_dt = 1.0 / 7.5  # average of actual measurements

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.display_msg('Initializing...')
        self.z = -40

    def reset(self):
        self.client.reset()
        self.setup_drones()

        ################################################################
        # # option 1: two flocks colliding
        # x0, v0 = twoflocks(self.n_agents)

        # initial_v_dt = 8.0  # good for twoflocks()
        # initial_v_dt = 2.0 # better for the rest of the cases

        # option 2: two circles with inwards velocities
        # x0, v0 = circle(N)

        # option 3: tight grid of agents
        # x0 = grid(N)

        # option 4: random initialization as in training
        #     states = problem.initialize()
        #     x0 = states[:,0:3]
        #     v0 = states[:,2:4]
        ######################################################################

        initial_v_dt = 4.0 
        x0 = grid(self.n_agents)
        bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        v0 = np.zeros((self.n_agents, 2))
        v0[:, 0] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0] 
        v0[:, 1] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1] 

        states = self.getStates()
        mean_x = np.mean(states[:, 0])
        mean_y = np.mean(states[:, 1])

        # scale positions and velocities
        x0 = x0 * self.scale
        v0 = v0 * self.scale

        self.display_msg('Moving to new positions...')
        self.send_loc_commands(x0, mean_x, mean_y)

        self.send_velocity_commands(v0, duration=initial_v_dt)

        self.x = self.getStates() / self.scale  # get drone locations and velocities
        self.compute_helpers()
        return (self.state_values, self.state_network)

    def step(self, u):
        # integrate acceleration
        new_vel = (u * self.true_dt + self.x[:, 2:4]) * self.scale
        self.send_velocity_commands(new_vel)
        self.x = self.getStates() / self.scale  # get drone locations and velocities
        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}


    def getStates(self):
        states = np.zeros(shape=(self.n_agents, 4))
        for i in range(0, self.n_agents):
            state = self.client.getMultirotorState(vehicle_name=self.names[i])
            states[i][0] = float(state.kinematics_estimated.position.x_val) + self.home[i][0]
            states[i][1] = float(state.kinematics_estimated.position.y_val) + self.home[i][1]
            states[i][2] = float(state.kinematics_estimated.linear_velocity.x_val)
            states[i][3] = float(state.kinematics_estimated.linear_velocity.y_val)
        return states


    def setup_drones(self):
        for i in range(0, self.n_agents):
            self.client.enableApiControl(True, self.names[i])
        for i in range(0, self.n_agents):
            self.client.armDisarm(True, self.names[i])

        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.takeoffAsync(vehicle_name=self.names[i]))  # .join()
        for f in fi:
            f.join()


    def send_velocity_commands(self, u, duration=0.01):
        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.moveByVelocityZAsync(u[i, 0], u[i, 1], self.z, duration, vehicle_name=self.names[i]))
        for f in fi:
            f.join()


    def send_loc_commands(self, loc, mean_x, mean_y):
        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.moveToPositionAsync(loc[i][0] - self.home[i][0] + mean_x, loc[i][1] - self.home[i][1] + mean_y, self.z, 6.0,
                                                 vehicle_name=self.names[i]))
        sleep(0.1)
        for f in fi:
            f._timeout = 10  # quads sometimes get stuck during a crash and never reach the destination
            f.join()


    def display_msg(self, msg):
        print(msg)
        self.client.simPrintLogMessage(msg)
