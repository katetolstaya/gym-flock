import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class FlockingEnv(gym.Env):

    def __init__(self):

        config_file = path.join(path.dirname(__file__), "params_flock.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = False # if the agents are moving or not
        self.mean_pooling = True # normalize the adjacency matrix by the number of neighbors or not

        # number states per agent
        self.nx_system = 4
        # numer of observations per agent
        self.n_features = 6
        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.n_agents = int(config['network_size'])
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max 
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt


        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        self.u = np.zeros((self.n_agents, self.nu))
        self.mean_vel = np.zeros((self.n_agents, self.nu))
        self.init_vel = np.zeros((self.n_agents, self.nu))
        self.a_net = np.zeros((self.n_agents, self.n_agents))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 
        self.gain = 10.0 # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_agents,),
                                       dtype=np.float32)


        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):

        #u = np.reshape(u, (-1, 2))
        assert u.shape == (self.n_agents, self.nu)
        self.u = u

        if self.dynamic:
            # x position
            self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt
            # y position
            self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.gain * self.u[:, 0] * self.dt #+ np.random.normal(0, self.std_dev, (self.n_agents,))
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.gain * self.u[:, 1] * self.dt #+ np.random.normal(0, self.std_dev, (self.n_agents,))

        return self._get_obs(), self.instant_cost(), False, {}

    def instant_cost(self):  # sum of differences in velocities
        # TODO adjust to desired reward
        # action_cost = -1.0 * np.sum(np.square(self.u))
         #curr_variance = -1.0 * np.sum((np.var(self.x[:, 2:4], axis=0)))
         versus_initial_vel = -1.0 * np.sum(np.sum(np.square(self.x[:, 2:4] - self.mean_vel), axis=1))
         #return curr_variance + versus_initial_vel
         return versus_initial_vel


    def reset(self):
        x = np.zeros((self.n_agents, self.nx_system))
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.1  # 0.25

        # generate an initial configuration with all agents connected,
        # and minimum distance between agents > min_dist_thresh
        while degree < 2 or min_dist < min_dist_thresh: 

            # randomly initialize the location and velocity of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_agents,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
            x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0]
            x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1]

            # compute distances between agents
            a_net = self.dist2_mat(x)

            # compute minimum distance between agents and degree of network to check if good initial configuration
            min_dist = np.sqrt(np.min(np.min(a_net)))
            a_net = a_net < self.comm_radius2
            degree = np.min(np.sum(a_net.astype(int), axis=1))

        # keep good initialization
        self.mean_vel = np.mean(x[:, 2:4], axis=0)
        self.init_vel = x[:, 2:4]
        self.x = x
        self.a_net = self.get_connectivity(self.x)
        return self._get_obs()

    def _get_obs(self):
        # state_values = self.x
        state_values = np.hstack((self.x, self.init_vel))  # initial velocities are part of state to make system observable
        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        return (state_values, state_network)


    def dist2_mat(self, x):
        """
        Compute squared euclidean distances between agents. Diagonal elements are infinity
        Args:
            x (): current state of all agents

        Returns: symmetric matrix of size (n_agents, n_agents) with A_ij the distance between agents i and j
        """
        x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net


    def get_connectivity(self, x):
        """
        Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
        Args:
            x (): current state of all agents

        Returns: adjacency matrix of network

        """
        a_net = self.dist2_mat(x)
        a_net = (a_net < self.comm_radius2).astype(float)

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors 

        return a_net

    def controller(self):
        """
        Consensus-based centralized flocking with no obstacle avoidance

        Returns: the optimal action
        """
        # TODO implement Tanner 2003?
        u = np.mean(self.x[:,2:4], axis=0) - self.x[:,2:4]
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        return u

    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            ax.plot([0], [0], 'kx')
            plt.ylim(-1.0 * self.r_max, 1.0 * self.r_max)
            plt.xlim(-1.0 * self.r_max, 1.0 * self.r_max)
            a = gca()
            a.set_xticklabels(a.get_xticks(), font)
            a.set_yticklabels(a.get_yticks(), font)
            plt.title('GNN Controller')
            self.fig = fig
            self.line1 = line1

        self.line1.set_xdata(self.x[:, 0])
        self.line1.set_ydata(self.x[:, 1])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        pass
 