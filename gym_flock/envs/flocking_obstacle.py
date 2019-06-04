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
        self.n_obstacles = 5
        
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
        self.x[0:self.n_obstacles,2:4] = 0
        return (self.state_values, self.state_network)

    def compute_helpers(self):

        self.diff = self.x.reshape((self.n_agents, 1, self.nx_system)) - self.x.reshape((1, self.n_agents, self.nx_system))

        # broken agents don't contribute to velocity differences
        self.diff[0:self.n_obstacles,:,2:4] = 0
        self.diff[:,0:self.n_obstacles,2:4] = 0


        self.r2 =  np.multiply(self.diff[:, :, 0], self.diff[:, :, 0]) + np.multiply(self.diff[:, :, 1], self.diff[:, :, 1])
        np.fill_diagonal(self.r2, np.Inf)

        self.adj_mat = (self.r2 < self.comm_radius2).astype(float)

        # Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
        n_neighbors = np.reshape(np.sum(self.adj_mat, axis=1), (self.n_agents,1)) # correct - checked this
        n_neighbors[n_neighbors == 0] = 1
        self.adj_mat_mean = self.adj_mat / n_neighbors 

        self.x_features = np.dstack((self.diff[:, :, 2], np.divide(self.diff[:, :, 0], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 0], self.r2),
                          self.diff[:, :, 3], np.divide(self.diff[:, :, 1], np.multiply(self.r2, self.r2)), np.divide(self.diff[:, :, 1], self.r2)))


        self.state_values = np.sum(self.x_features * self.adj_mat.reshape(self.n_agents, self.n_agents, 1), axis=1)
        self.state_values = self.state_values.reshape((self.n_agents, self.n_features))

        if self.mean_pooling:
            self.state_network = self.adj_mat_mean
        else:
            self.state_network = self.adj_mat
