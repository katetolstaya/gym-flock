import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
# from scipy.spatial.distance import pdist, squareform
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

        self.fig = None
        self.line1 = None
        self.nx_system = 4
        self.n_nodes = int(config['network_size'])
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max  # 0.5 * self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # number of features and outputs
        self.n_features = int(config['N_features'])
        self.nu = int(config['N_outputs'])  # outputs

        self.x = np.zeros((self.n_nodes, self.nx_system))
        self.u = np.zeros((self.n_nodes, self.nu))
        self.mean_vel = np.zeros((self.n_nodes, self.nu))
        self.init_vel = np.zeros((self.n_nodes, self.nu))

        # TODO
        self.max_accel = 1
        self.max_z = 200

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 * self.n_nodes,),
                                       dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.max_z, high=self.max_z, shape=(self.n_features * self.n_nodes,),
                                            dtype=np.float32)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x = self.x
        x_ = np.zeros((self.n_nodes, self.nx_system))
        u = np.reshape(u, (-1, 2))

        # u = np.vstack((np.zeros((self.n_leaders, 2)), u))
        # x position
        x_[:, 0] = x[:, 0] + x[:, 2] * self.dt
        # y position
        x_[:, 1] = x[:, 1] + x[:, 3] * self.dt
        # x velocity
        x_[:, 2] = x[:, 2] + u[:, 0] * self.dt + np.random.normal(0, self.std_dev, (self.n_nodes,))
        # y velocity
        x_[:, 3] = x[:, 3] + u[:, 1] * self.dt + np.random.normal(0, self.std_dev, (self.n_nodes,))
        # TODO - check the 0.1
        self.x = x_
        self.u = u

        #return (self._get_obs(), self.costs_list()), self.instant_cost(), False, {}
        return self._get_obs(), self.instant_cost(), False, {}

    def instant_cost(self):  # sum of differences in velocities
        # return np.sum(np.var(self.x[:, 2:4], axis=0))  # + np.sum(np.square(self.u)) * 0.00001

        s_costs = -1.0 * np.sum(np.square(self.x[:, 2:4] - self.mean_vel), axis=1)
        return np.sum(s_costs) #+ np.sum(np.square(self.u)) # todo add an action cost

    def _get_obs(self):
        return (np.hstack((self.x, self.init_vel)), self.get_connectivity(self.x))


    def reset(self):
        x = np.zeros((self.n_nodes, self.nx_system))
        degree = 0
        min_dist = 0

        while degree < 2 or min_dist < 0.1:  # < 0.25:  # 0.25:  #0.5: #min_dist < 0.25:
            # randomly initialize the state of all agents
            length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_nodes,)))
            angle = np.pi * np.random.uniform(0, 2, size=(self.n_nodes,))
            x[:, 0] = length * np.cos(angle)
            x[:, 1] = length * np.sin(angle)

            bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
            x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_nodes,)) + bias[0]
            x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_nodes,)) + bias[1]

            # compute distances between agents
            x_t_loc = x[:, 0:2]  # x,y location determines connectivity

            a_net = np.sqrt(
                np.sum(np.square(x_t_loc.reshape((self.n_nodes, 1, 2)) - x_t_loc.reshape((1, self.n_nodes, 2))),
                       axis=2))

            # no self loops
            a_net = a_net + 2 * self.comm_radius * np.eye(self.n_nodes)

            # compute minimum distance between agents and degree of network
            min_dist = np.min(np.min(a_net))
            a_net = a_net < self.comm_radius
            degree = np.min(np.sum(a_net.astype(int), axis=1))

            self.mean_vel = np.mean(x[:, 2:4], axis=0)
            self.init_vel = x[:, 2:4]

        self.x = x

        return self._get_obs()

    def close(self):
        pass


    def get_connectivity(self, x):
        """
        Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
        Args:
            x (): current states of all agents

        Returns: adjacency matrix of network

        """
        x_t_loc = x[:, 0:2]  # x,y location determines connectivity
        # a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))
        a_net = np.sum(np.square(x_t_loc.reshape((self.n_nodes, 1, 2)) - x_t_loc.reshape((1, self.n_nodes, 2))), axis=2)
        a_net = (a_net < self.comm_radius2).astype(float)
        # TODO normalize
        np.fill_diagonal(a_net, 0)
        return a_net

    def controller(self):
        """
        The controller for flocking from Turner 2003.
        Args:
            x (): the current state
        Returns: the optimal action
        """
        mean_vel = np.mean(self.x[:,2:4], axis=0)
        u = mean_vel - self.x[:,2:4]
        u = u * 10
        u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        return u

    def render(self, mode='human'):

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
 