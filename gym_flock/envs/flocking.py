import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
from scipy.spatial.distance import pdist, squareform
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca

font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 14}

class FlockingEnv(gym.Env):

    def __init__(self):

        config_file = path.join(path.dirname(__file__), "params_flock.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.n_leaders = 5

        self.fig = None
        self.line1 = None
        self.filter_len = int(config['filter_length'])
        self.nx_system = 4
        self.n_nodes = int(config['network_size']) + self.n_leaders
        self.comm_radius = float(config['comm_radius'])
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max  # 0.5 * self.v_max
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        self.pooling = []
        if config.getboolean('sum_pooling'):
            self.pooling.append(np.nansum)
        if config.getboolean('min_pooling'):
            self.pooling.append(np.nanmin)
        if config.getboolean('max_pooling'):
            self.pooling.append(np.nanmax)
        self.n_pools = len(self.pooling)

        # number of features and outputs
        self.n_features = int(config['N_features'])
        self.nx = int(self.n_features / self.n_pools / self.filter_len)
        self.nu = int(config['N_outputs'])  # outputs

        self.x_agg = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        self.x = np.zeros((self.n_nodes, self.nx_system))
        self.u = np.zeros((self.n_nodes, self.nu))

        # TODO
        self.max_accel = 40
        self.max_z = 200  

        self.b = np.ones((self.n_nodes,1))

        # self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(self.n_nodes, 2), dtype=np.float32 )
        # self.observation_space = spaces.Box(low=-self.max_z, high=self.max_z, shape=(
        # self.n_nodes, self.nx * self.filter_len * self.n_pools) , dtype=np.float32)

        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2,) , dtype=np.float32 )
        self.observation_space = spaces.Box(low=-self.max_z, high=self.max_z, shape=(self.n_features, ), dtype=np.float32)

        self.seed()

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


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, u):
        x = self.x
        x_ = np.zeros((self.n_nodes, self.nx_system))

        u = np.vstack((np.zeros((self.n_leaders, 2)), u))
        # x position
        x_[:, 0] = x[:, 0] + x[:, 2] * self.dt
        # y position
        x_[:, 1] = x[:, 1] + x[:, 3] * self.dt
        # x velocity
        x_[:, 2] = x[:, 2] + self.b * 0.1 * u[:, 0] * self.dt + np.random.normal(0, self.std_dev,(self.n_nodes,))
        # y velocity
        x_[:, 3] = x[:, 3] + self.b * 0.1 * u[:, 1] * self.dt + np.random.normal(0, self.std_dev,(self.n_nodes,))
        # TODO - check the 0.1
        self.x = x_
        self.x_agg = self.aggregate(self.x, self.x_agg)
        self.u = u

        return self._get_obs(), -self.instant_cost(), False, {}

    def instant_cost(self):  # sum of differences in velocities
        return np.sum(np.var(self.x[:, 2:4], axis=0)) #+ np.sum(np.square(self.u)) * 0.00001

    def _get_obs(self):
        reshaped = self.x_agg.reshape((self.n_nodes, self.n_features))
        clipped = np.clip(reshaped, a_min=-self.max_z, a_max=self.max_z)
        return clipped[self.n_leaders:, :]

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

            x[0:2, 2] = bias[0]
            x[0:2, 3] = bias[1]

            # compute distances between agents
            x_t_loc = x[:, 0:2]  # x,y location determines connectivity
            a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))

            # no self loops
            a_net = a_net + 2 * self.comm_radius * np.eye(self.n_nodes)

            # compute minimum distance between agents and degree of network
            min_dist = np.min(np.min(a_net))
            a_net = a_net < self.comm_radius
            degree = np.min(np.sum(a_net.astype(int), axis=1))


        # the first two agents are the leaders
        self.b = np.ones((self.n_nodes,))
        self.b[0] = 0
        self.b[1] = 0
        # self.b[np.argmax(np.linalg.norm(x[:,2:4], axis=1))] = 0

        self.x = x
        self.x_agg = np.zeros((self.n_nodes, self.nx * self.filter_len, self.n_pools))
        self.x_agg = self.aggregate(self.x, self.x_agg)

        return self._get_obs()

    # def render(self, mode='human'):
    #     pass

    def close(self):
        pass

    def aggregate(self, xt, x_agg):
        """
        Perform aggegration operation for all possible pooling operations using helper functions get_pool and get_comms
        Args:
            x_agg (): Last time step's aggregated info
            xt (): Current state of all agents

        Returns:
            Aggregated state values
        """

        x_features = self.get_x_features(xt)
        a_net = self.get_connectivity(xt)
        for k in range(0, self.n_pools):
            comm_data = self.get_comms(np.dstack((x_features, self.get_features(x_agg[:, :, k]))), a_net)
            x_agg[:, :, k] = self.get_pool(comm_data, self.pooling[k])
        return x_agg

    def get_connectivity(self, x):
        """
        Get the adjacency matrix of the network based on agent locations by computing pairwise distances using pdist
        Args:
            x (): current states of all agents

        Returns: adjacency matrix of network

        """
        x_t_loc = x[:, 0:2]  # x,y location determines connectivity
        a_net = squareform(pdist(x_t_loc.reshape((self.n_nodes, 2)), 'euclidean'))
        a_net = (a_net < self.comm_radius).astype(float)
        np.fill_diagonal(a_net, 0)
        return a_net

    def get_x_features(self, xt):
        """
        Compute the non-linear features necessary for implementing Turner 2003
        Args:
            xt (): current state of all agents

        Returns: matrix of features for each agent

        """

        diff = xt.reshape((self.n_nodes, 1, self.nx_system)) - xt.reshape((1, self.n_nodes, self.nx_system))
        r2 = np.multiply(diff[:, :, 0], diff[:, :, 0]) + np.multiply(diff[:, :, 1], diff[:, :, 1]) + np.eye(
            self.n_nodes)
        return np.dstack((diff[:, :, 2], np.divide(diff[:, :, 0], np.multiply(r2, r2)), np.divide(diff[:, :, 0], r2),
                          diff[:, :, 3], np.divide(diff[:, :, 1], np.multiply(r2, r2)), np.divide(diff[:, :, 1], r2)))

    def get_features(self, agg):
        """
        Matrix of
        Args:
            agg (): the aggregated matrix from the last time step

        Returns: matrix of aggregated features from all nodes at current time

        """
        return np.tile(agg[:, :-self.nx].reshape((self.n_nodes, 1, -1)), (1, self.n_nodes, 1))  # TODO check indexing

    def get_comms(self, mat, a_net):
        """
        Enforces that agents who are not connected in the network cannot observe each others' states
        Args:
            mat (): matrix of state information for the whole graph
            a_net (): adjacency matrix for flock network (weighted networks unsupported for now)

        Returns:
            mat (): sparse matrix with NaN values where agents can't communicate

        """
        a_net[a_net == 0] = np.nan
        return mat * a_net.reshape(self.n_nodes, self.n_nodes, 1)

    def get_pool(self, mat, func):
        """
        Perform pooling operations on the matrix of state information. The replacement of values with NaNs for agents who
        can't communicate must already be enforced.
        Args:
            mat (): matrix of state information
            func (): pooling function (np.nansum(), np.nanmin() or np.nanmax()). Must ignore NaNs.

        Returns:
            information pooled from neighbors for each agent

        """
        return func(mat, axis=1).reshape((self.n_nodes, self.n_features))  # TODO check this axis = 1

