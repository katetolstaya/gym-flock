import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from sklearn.neighbors import NearestNeighbors
import itertools 
import random
import pdb
font = {'family': 'sans-serif',
        'weight': 'bold',
        'size': 14}


class FormationFlyingEnv(gym.Env):

    def __init__(self):

        config_file = path.join(path.dirname(__file__), "formation_flying.cfg")
        config = configparser.ConfigParser()
        config.read(config_file)
        config = config['flock']

        self.dynamic = True # if the agents are moving or not
        self.mean_pooling = False # normalize the adjacency matrix by the number of neighbors or not
        #self.degree =  4 # number of nearest neighbors (if 0, use communication range instead)
        self.degree = 1 
        # number states per agent
        self.nx_system = 4
        # numer of features per agent
        self.n_features = 4
        # number of actions per agent
        self.nu = 2 

        # problem parameters from file
        self.n_agents = 3
        self.comm_radius = float(config['comm_radius'])
        self.comm_radius2 = self.comm_radius * self.comm_radius
        self.dt = float(config['system_dt'])
        self.v_max = float(config['max_vel_init'])
        self.v_bias = self.v_max 
        self.r_max = float(config['max_rad_init'])
        self.std_dev = float(config['std_dev']) * self.dt

        # intitialize state matrices
        self.x = np.zeros((self.n_agents, self.nx_system))
        
        self.a_net = np.zeros((self.n_agents, self.n_agents))

        # TODO : what should the action space be? is [-1,1] OK?
        self.max_accel = 1 
        self.gain = 1.0 # TODO - adjust if necessary - may help the NN performance
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2 *self.n_agents,),dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.Inf, high=np.Inf, shape=(self.n_agents, self.n_features),
                                            dtype=np.float32)

        self.fig = None
        self.line1 = None

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
             
        self.u = np.reshape(action,(self.n_agents, self.nu))
        
        
        self.x[:, 0] = self.x[:, 0] + self.u[:, 0]*0.1
        # update  y position 
        self.x[:, 1] = self.x[:, 1] + self.u[:, 1]*0.1

        return self._get_obs(), self.instant_cost(), False, {}

    def instant_cost(self):  # sum of differences in velocities
       
        robot_xs = self.x[:,0]
        robot_ys = self.x[:,1]

        robot_goalxs = self.x[:,2]
        robot_goalys = self.x[:,3]

        diff = (robot_xs - robot_goalxs)**2 + (robot_ys - robot_goalys)**2
        return -np.sum(diff)


    def reset(self):
        x = np.zeros((self.n_agents, self.n_features)) #+2 to account for goal x and goal y
        degree = 0
        min_dist = 0
        min_dist_thresh = 0.01  # 0.25

        #set arbitrary goal 
        self.goal_x1 = 0
        self.goal_y1 = 2

        self.goal_x2 = -2
        self.goal_y2 = 2

        self.goal_x3 = 2
        self.goal_y3 = 2
        

        xpoints = np.array((0,-2,2))
        ypoints = np.array((0,0,0))


        self.start_xpoints = xpoints
        self.start_ypoints = ypoints

        self.goal_xpoints = np.array((self.goal_x1,self.goal_x2,self.goal_x3))
        self.goal_ypoints = np.array((self.goal_y1,self.goal_y2,self.goal_y3))
        
        x[:,0] = xpoints
        x[:,1] = ypoints

        
        x[:,2] = np.array((self.goal_x1,self.goal_x2,self.goal_x3))
        x[:,3] = np.array((self.goal_y1,self.goal_y2,self.goal_y3))
        
        # compute distances between agents
        a_net = self.dist2_mat(x)

        # compute minimum distance between agents and degree of network to check if good initial configuration
        min_dist = np.sqrt(np.min(np.min(a_net)))
        a_net = a_net < self.comm_radius2
        degree = np.min(np.sum(a_net.astype(int), axis=1))

        self.x = x

        self.a_net = self.get_connectivity(self.x)

        return self._get_obs()

    def _get_obs(self):
        
        state_values = self.x   
        if self.dynamic:
            state_network = self.get_connectivity(self.x)
        else:
            state_network = self.a_net

        #return (state_values, state_network)
        return state_values

    def dist2_mat(self, x):

        x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        np.fill_diagonal(a_net, np.Inf)
        return a_net


    def get_connectivity(self, x):


        if self.degree == 0:
            a_net = self.dist2_mat(x)
            a_net = (a_net < self.comm_radius2).astype(float)
        else:
            neigh = NearestNeighbors(n_neighbors=self.degree)
            neigh.fit(x[:,2:4])
            a_net = np.array(neigh.kneighbors_graph(mode='connectivity').todense())

        if self.mean_pooling:
            # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
            n_neighbors = np.reshape(np.sum(a_net, axis=1), (self.n_agents,1)) # TODO or axis=0? Is the mean in the correct direction?
            n_neighbors[n_neighbors == 0] = 1
            a_net = a_net / n_neighbors 

        return a_net


    def render(self, mode='human'):
        """
        Render the environment with agents as points in 2D space
        """

        if self.fig is None:
            plt.ion()
            fig = plt.figure()
            ax = fig.add_subplot(111)
            line1, = ax.plot(self.x[:, 0], self.x[:, 1], 'bo')  # Returns a tuple of line objects, thus the comma
            
            #ax.plot([0], [0], 'kx')
            ax.plot(self.start_xpoints, self.start_ypoints, 'kx')
            ax.plot(self.goal_xpoints, self.goal_ypoints, 'rx')

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
 