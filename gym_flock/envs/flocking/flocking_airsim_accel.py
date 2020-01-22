import airsim
import numpy as np
from time import sleep
from gym_flock.envs.flocking.flocking_relative import FlockingRelativeEnv
from gym_flock.envs.flocking.utils import grid, parse_settings


class FlockingAirsimAccelEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingAirsimAccelEnv, self).__init__()

        # parse settings file with drone names and home locations
        fname = '/home/kate/Documents/AirSim/settings.json'
        self.names, self.home = parse_settings(fname)
        self.n_agents = len(self.names)

        # rescale locations and velocities to avoid changing potential function
        self.scale = 6.0

        # duration of velocity commands
        # self.true_dt = 1.0 / 7.5  # average of actual measurements

        # connect to the AirSim simulator
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        # self.display_msg('Initializing...')
        self.z = -50
        self.yaws = None
        self.max_accel = 0.5

    def reset(self):
        self.client.reset()
        self.setup_drones()

        ################################################################
        # # option 1: two flocks colliding
        # x0, v0 = twoflocks_old(self.n_agents)
        #
        # initial_v_dt = 8.0  # good for twoflocks()
        #initial_v_dt = 2.0 # better for the rest of the cases

        # option 2: two circles with inwards velocities
        # x0, v0 = circle(N)

        # option 3: tight grid of agents
        # x0 = grid(N)

        # option 4: random initialization as in training
        #     states = problem.initialize()
        #     x0 = states[:,0:3]
        #     v0 = states[:,2:4]
        ######################################################################

        initial_v_dt = 2.0
        x0 = grid(self.n_agents)
        bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        v0 = np.zeros((self.n_agents, 2))
        self.v_max = 1.0
        v0[:, 0] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0]
        v0[:, 1] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1]

        states, self.yaws = self.get_states()
        mean_x = np.mean(states[:, 0])
        mean_y = np.mean(states[:, 1])

        # scale positions and velocities
        x0 = x0 * self.scale
        v0 = v0 * self.scale

        # self.display_msg('Moving to new positions...')
        self.send_loc_commands(x0, mean_x, mean_y)

        self.send_velocity_commands(v0, duration=initial_v_dt)
        states, self.yaws = self.get_states()
        self.x = states / self.scale  # get drone locations and velocities
        self.compute_helpers()
        return (self.state_values, self.state_network)

    def step(self, u):

        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        u = u * self.scale

        # roll = u[:, 1] / 9.8
        # pitch = -1.0 * u[:, 0] / 9.8

        # TODO test this: (is correct when yaw is pi/2)?
        roll = (u[:, 1] * np.cos(self.yaws) - u[:, 0] * np.sin(self.yaws)) / 9.8
        pitch = (-1.0 * u[:, 0] * np.cos(self.yaws) - 1.0 * u[:, 1] * np.sin(self.yaws))/9.8

        roll_pitch = np.hstack((pitch.reshape((-1,1)), roll.reshape((-1,1))))
        self.send_accel_commands(roll_pitch)

        states, self.yaws = self.get_states()
        self.x = states / self.scale  # get drone locations and velocities
        self.compute_helpers()
        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def quaternion_to_yaw(self, q):
        # yaw (z-axis rotation) from quaternion
        w = float(q.w_val)
        x = float(q.x_val)
        y = float(q.y_val)
        z = float(q.z_val)
        siny_cosp = +2.0 * (w * z + x * y)
        cosy_cosp = +1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return yaw

    def get_states(self):
        states = np.zeros(shape=(self.n_agents, 4))
        yaws = np.zeros(shape=(self.n_agents, 1))
        for i in range(0, self.n_agents):
            state = self.client.getMultirotorState(vehicle_name=self.names[i])
            states[i][0] = float(state.kinematics_estimated.position.x_val) + self.home[i][0]
            states[i][1] = float(state.kinematics_estimated.position.y_val) + self.home[i][1]
            states[i][2] = float(state.kinematics_estimated.linear_velocity.x_val)
            states[i][3] = float(state.kinematics_estimated.linear_velocity.y_val)

            yaws[i] = self.quaternion_to_yaw(state.kinematics_estimated.orientation)

        return states, yaws

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

    def send_accel_commands(self, u, duration=0.01):
        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.moveByAngleZAsync(float(u[i, 0]), float(u[i, 1]), self.z, 0.0, duration, vehicle_name=self.names[i]))
        for f in fi:
            f.join()

    def send_velocity_commands(self, u, duration=0.01):
        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.moveByVelocityZAsync(float(u[i, 0]), float(u[i, 1]), self.z, duration, vehicle_name=self.names[i]))
        for f in fi:
            f.join()

    def send_loc_commands(self, loc, mean_x, mean_y):
        fi = []
        for i in range(self.n_agents):
            fi.append(self.client.moveToPositionAsync(loc[i][0] - self.home[i][0] + mean_x,
                                                      loc[i][1] - self.home[i][1] + mean_y, self.z, 6.0,
                                                      vehicle_name=self.names[i]))
        sleep(0.1)
        for f in fi:
            f._timeout = 10  # quads sometimes get stuck during a crash and never reach the destination
            f.join()

    def display_msg(self, msg):
        print(msg)
        self.client.simPrintLogMessage(msg)

    def controller(self, centralized=None):
        """
        The controller for flocking from Turner 2003.
        Returns: the optimal action
        """
        controls = super(FlockingAirsimAccelEnv, self).controller(centralized)
        controls = np.clip(controls, -1.0 * self.max_accel, self.max_accel)
        return controls

