
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from numpy import linalg as LA


class GG:

    def __init__(self, learning, inputsize, sigma, lambda_, gamma, softmax_exponent, nbr_input, max_epoch):

        self.name = 'GG'
        self.learning = learning
        self.inputsize = inputsize
        self.sigma = sigma
        self.lambda_ = lambda_  # adaptation step
        self.gamma = gamma      # tunning step
        self.softmax_exponent = softmax_exponent
        self.nbr_input = nbr_input
        self.max_epoch = max_epoch
        self.metric = 'Euclidean'
        self.outputsize_x = 2
        self.outputsize_y = 2
        self.nbr_neurons = self.outputsize_x * self.outputsize_y
        self.normalize_input = False
        self.normalize_weights = False
        self.softmax_normalization = True
        self.ft_phase = False
        self.learningRate = 0.1
        self.learningRate_decay = 0.9999
        self.learningRate_min = 0.01
        self.cnt_input = 0
        self.cnt_lambda = 0
        self.win_counter = np.zeros((2, 2))
        self.weights = np.zeros((2, 2, inputsize))  # Rows, Columns, Depth
        self.activity_pattern = []
        self.ft_epoch = 0

    def normalize(self, state):

        if self.normalize_input:
            state /= LA.norm(np.expand_dims(state, axis=0))

        return state

    def soft_max_normalization(self, state):

        m = np.max(state)
        if m != 0:
            state /= m

        return state

    def set_activity(self, state):

        if self.metric == 'Euclidean':
            dist = np.sum((state - self.weights) ** 2, axis=2)
            activity = np.exp(-dist / self.sigma)

        else:
            # Scalar Product
            mat_mul = state * self.weights
            activity = mat_mul.sum(axis=2)

        if self.softmax_exponent != 1:
            activity = activity ** self.softmax_exponent

        if self.softmax_normalization:
            activity = self.soft_max_normalization(activity)

        return activity

    def find_winning_node(self, activity):

        winner_x, winner_y = np.unravel_index(np.argmax(activity, axis=None), activity.shape)
        winning_node = np.array([winner_x, winner_y])

        return winning_node

    def learn(self, state, winner):

        err = state - self.weights
        param = 0.8

        # Update winner
        self.weights[winner[0], winner[1], :] += self.learningRate * err[winner[0], winner[1], :]

        # Update neighborhood#1
        if winner[0] - 1 in range(0, self.outputsize_x):
            self.weights[winner[0] - 1, winner[1], :] += param * self.learningRate * err[winner[0] - 1, winner[1], :]

        # Update neighborhood#2
        if winner[0] + 1 in range(0, self.outputsize_x):
            self.weights[winner[0] + 1, winner[1], :] += param * self.learningRate * err[winner[0] + 1, winner[1], :]

        # Update neighborhood#3
        if winner[1] - 1 in range(0, self.outputsize_y):
            self.weights[winner[0], winner[1] - 1, :] += param * self.learningRate * err[winner[0], winner[1] - 1, :]

        # Update neighborhood#4
        if winner[1] + 1 in range(0, self.outputsize_y):
            self.weights[winner[0], winner[1] + 1, :] += param * self.learningRate * err[winner[0], winner[1] + 1, :]

        # Learning decay
        if self.ft_phase:
            self.learningRate = self.learningRate_decay * self.learningRate
            if self.learningRate < self.learningRate_min:
                self.learningRate = self.learningRate_min

    def get_distance_to_neighbors(self, abs_winner):
        """
            This function Calculates the distance to the 4 direct topological neighbors of the absolute winner

                Args:
                    abs_winner: The neuron activated the most during adaptation phase

                Returns:
                    dist: distance array of four values
                """

        dist = -1 * np.ones((1, 4))

        # distance to neighbor #1
        if abs_winner[0] - 1 in range(0, self.outputsize_x):
            dist[0, 0] = LA.norm(
                self.weights[abs_winner[0], abs_winner[1], :] - self.weights[abs_winner[0] - 1, abs_winner[1], :])

        # distance to neighbor #2
        if abs_winner[0] + 1 in range(0, self.outputsize_x):
            dist[0, 1] = LA.norm(
                self.weights[abs_winner[0], abs_winner[1], :] - self.weights[abs_winner[0] + 1, abs_winner[1], :])

        # distance to neighbor #3
        if abs_winner[1] - 1 in range(0, self.outputsize_y):
            dist[0, 2] = LA.norm(
                self.weights[abs_winner[0], abs_winner[1], :] - self.weights[abs_winner[0], abs_winner[1] - 1, :])

        # distance to neighbor #4
        if abs_winner[1] + 1 in range(0, self.outputsize_y):
            dist[0, 3] = LA.norm(
                self.weights[abs_winner[0], abs_winner[1], :] - self.weights[abs_winner[0], abs_winner[1] + 1, :])

        return dist

    def insert_new_neurons(self, abs_winner, nei_y):
        """
            This function inserts a complete row/column between the absolute winner and its furthest direct topological
            neighbor

                Args:
                    abs_winner: The neuron activated the most during adaptation phase
                    nei_y: The index of the furthest direct topological neighbor to the absolute winner
                            0: insert a row to the top of the absolute winner
                            1: insert a row to the bottom of the absolute winner
                            2: insert a column to the left of the absolute winner
                            3: insert a column to the right of the absolute winner

                """

        if nei_y == 0:
            weight_n = 0.5 * (self.weights[abs_winner[0], :, :] + self.weights[abs_winner[0] - 1, :, :])
            weight_n = np.expand_dims(weight_n, 0)
            weights_p = np.concatenate((self.weights[0:abs_winner[0], :, :], weight_n), axis=0)
            self.weights = np.concatenate((weights_p,
                                           self.weights[abs_winner[0]:self.outputsize_x + 1, :, :]),
                                          axis=0)

        if nei_y == 1:
            weight_n = 0.5 * (self.weights[abs_winner[0], :, :] + self.weights[abs_winner[0] + 1, :, :])
            weight_n = np.expand_dims(weight_n, 0)
            weights_p = np.concatenate((self.weights[0:abs_winner[0] + 1, :, :], weight_n), axis=0)
            self.weights = np.concatenate((weights_p,
                                           self.weights[abs_winner[0] + 1:self.outputsize_x + 1, :, :]),
                                          axis=0)

        if nei_y == 2:
            weight_n = 0.5 * (self.weights[:, abs_winner[1], :] + self.weights[:, abs_winner[1] - 1, :])
            weight_n = np.expand_dims(weight_n, 1)
            weights_p = np.concatenate((self.weights[:, 0:abs_winner[1], :], weight_n), axis=1)
            self.weights = np.concatenate((weights_p,
                                           self.weights[:, abs_winner[1]:self.outputsize_y + 1, :]),
                                          axis=1)

        if nei_y == 3:
            weight_n = 0.5 * (self.weights[:, abs_winner[1], :] + self.weights[:, abs_winner[1] + 1, :])
            weight_n = np.expand_dims(weight_n, 1)
            weights_p = np.concatenate((self.weights[:, 0:abs_winner[1] + 1, :], weight_n), axis=1)
            self.weights = np.concatenate((weights_p,
                                           self.weights[:, abs_winner[1] + 1:self.outputsize_y + 1, :]),
                                          axis=1)

    def run_growth_phase(self, winner):

        self.cnt_lambda += 1
        self.win_counter[winner[0]][winner[1]] += 1

        # ------------ Inserting new neurons
        if self.cnt_lambda == self.lambda_:
            # Get the absolute winner
            abs_winner = self.find_winning_node(self.win_counter)

            # Get distances to the 4 direct topological neighbors of the absolute winner
            dist = self.get_distance_to_neighbors(abs_winner)

            # Find the neighbor with the maximum distance to the absolute winner
            nei_x, nei_y = np.unravel_index(np.argmax(dist, axis=None), dist.shape)

            # Insert the new row/column
            self.insert_new_neurons(abs_winner, nei_y)

            # reset system parameters
            self.outputsize_x = np.size(self.weights, 0)
            self.outputsize_y = np.size(self.weights, 1)
            self.win_counter = np.zeros((self.outputsize_x, self.outputsize_y))
            self.cnt_lambda = 0

        self.cnt_input += 1
        # One epoch terminates: reset system parameters
        if self.cnt_input == self.nbr_input:
            self.cnt_input = 0
            self.win_counter = np.zeros((self.outputsize_x, self.outputsize_y))
            self.cnt_lambda = 0

    def growing_grid(self, input_data):

        self.outputsize_x = np.size(self.weights, 0)
        self.outputsize_y = np.size(self.weights, 1)
        self.nbr_neurons = self.outputsize_x * self.outputsize_y

        input_data = np.expand_dims(input_data, 0)
        input_data = self.normalize(input_data)

        # Set the network activity map using a metric
        activity = self.set_activity(input_data)

        # Find the Winner
        winner = self.find_winning_node(activity)

        if self.learning:
            self.learn(input_data, winner)

        if self.nbr_neurons < self.gamma:
            self.run_growth_phase(winner)

        else:

            self.ft_phase = True

        return activity, winner
