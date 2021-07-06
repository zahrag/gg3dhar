
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import pickle
import os


def save_models(path, model, save=None):

    if save:
        os.getcwd()
        with open(path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path):

    with open(path, 'rb') as handle:
        model = pickle.load(handle)

    return model
