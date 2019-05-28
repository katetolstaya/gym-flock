import gym
from gym import spaces, error, utils
from gym.utils import seeding
import numpy as np
import configparser
from os import path
import matplotlib.pyplot as plt
from matplotlib.pyplot import gca
from gym_flock.envs.flocking_relative import FlockingRelativeEnv

class FlockingObstacleEnv(FlockingRelativeEnv):

    def __init__(self):

        super(FlockingObstacleEnv, self).__init__()
        self.n_obstacles = 3
        
        self.mask = np.ones((self.n_agents,))
        self.mask[0:self.n_obstacles] = 0


    def params_from_cfg(self, args):
        super(FlockingObstacleEnv, self).params_from_cfg(args)
        self.mask[0:self.n_obstacles] = 0

    def step(self, u):

        #u = np.reshape(u, (-1, 2))
        assert u.shape == (self.n_agents, self.nu)
        #u = np.clip(u, a_min=-self.max_accel, a_max=self.max_accel)
        self.u = u

        # x position
        self.x[:, 0] = self.x[:, 0] + self.x[:, 2] * self.dt 
        # y position
        self.x[:, 1] = self.x[:, 1] + self.x[:, 3] * self.dt 
        # x velocity
        self.x[:, 2] = self.x[:, 2] + self.gain * self.u[:, 0] * self.dt * self.mask 
        # y velocity
        self.x[:, 3] = self.x[:, 3] + self.gain * self.u[:, 1] * self.dt * self.mask

        self.compute_helpers()

        return (self.state_values, self.state_network), self.instant_cost(), False, {}

    def reset(self):
        super(FlockingObstacleEnv, self).reset()
        # x = np.zeros((self.n_agents, self.nx_system))
        # degree = 0
        # min_dist = 0
        # min_dist_thresh = 0.1  # 0.25

        # # generate an initial configuration with all agents connected,
        # # and minimum distance between agents > min_dist_thresh
        # while degree < 2 or min_dist < min_dist_thresh: 

        #     # randomly initialize the location and velocity of all agents
        #     length = np.sqrt(np.random.uniform(0, self.r_max, size=(self.n_agents,)))
        #     angle = np.pi * np.random.uniform(0, 2, size=(self.n_agents,))
        #     x[:, 0] = length * np.cos(angle)
        #     x[:, 1] = length * np.sin(angle)


        #     bias = np.random.uniform(low=-self.v_bias, high=self.v_bias, size=(2,))
        #     x[:, 2] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[0] 
        #     x[:, 3] = np.random.uniform(low=-self.v_max, high=self.v_max, size=(self.n_agents,)) + bias[1] 

        #     # compute distances between agents
        #     x_loc = np.reshape(x[:, 0:2], (self.n_agents,2,1))
        #     a_net = np.sum(np.square(np.transpose(x_loc, (0,2,1)) - np.transpose(x_loc, (2,0,1))), axis=2)
        #     np.fill_diagonal(a_net, np.Inf)

        #     # compute minimum distance between agents and degree of network to check if good initial configuration
        #     min_dist = np.sqrt(np.min(np.min(a_net)))
        #     a_net = a_net < self.comm_radius2
        #     degree = np.min(np.sum(a_net.astype(int), axis=1))

        # # keep good initialization
        # self.mean_vel = np.mean(x[:, 2:4], axis=0)
        # self.init_vel = x[:, 2:4]
        # self.x = x
        # #self.a_net = self.get_connectivity(self.x)
        # self.compute_helpers()

        self.x[0:self.n_obstacles,2:4] = 0

        return (self.state_values, self.state_network)
