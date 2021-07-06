
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from numpy import linalg as LA
import math


class SNN:
    def __init__(self, learning, outputsize_x, outputsize_y, inputsize):

        self.name = 'SNN'
        self.learning = learning
        self.outputsize_x = outputsize_x
        self.outputsize_y = outputsize_y
        self.inputsize = inputsize
        self.normalize_input = True
        self.beta = 0.35
        self.weights = np.random.rand(outputsize_x, outputsize_y, inputsize)  # Rows, Columns, Depth

        # Normalize native weights
        w_n = LA.norm(self.weights, axis=2)
        self.weights /= np.expand_dims(w_n, axis=2)

    def normalize(self, state):

        if self.normalize_input and LA.norm(state) != 0:
            state /= LA.norm(np.expand_dims(state, axis=0))

        return state

    def set_activity(self, state):

        mat_mul = self.weights * state
        activity = mat_mul.sum(axis=2)

        return activity

    def find_winning_neuron(self, activity):

        winner_x, winner_y = np.unravel_index(np.argmax(activity, axis=None), activity.shape)

        return winner_x, winner_y

    def learn(self, input_data, activity, desired_output):

        if self.learning:
            err = desired_output - activity
            err = np.expand_dims(err, axis=2)
            self.weights += self.beta * input_data * err

            # normalize weights
            w_n = LA.norm(self.weights, axis=2)
            self.weights /= np.expand_dims(w_n, axis=2)

    def run_SNN(self, input_data, index_class):

        # get the correct action
        desired_output = np.zeros((self.outputsize_x, 1))
        desired_output[index_class, 0] = 1.0
        max_desired_x, max_desired_y = self.find_winning_neuron(desired_output)

        # normalize input
        input_data = self.normalize(np.expand_dims(input_data, axis=0))

        # set the activity
        activity = self.set_activity(np.expand_dims(input_data, axis=0))

        # get the system chosen action
        max_activity_x, max_activity_y = self.find_winning_neuron(activity)

        # compare and set the recognition result
        if max_activity_x == max_desired_x and max_activity_y == max_desired_y:
            result = 1.0
        else:
            result = 0.0

        # learning
        self.learn(input_data, activity, desired_output)

        # print('\n activity', activity)

        return activity, result

