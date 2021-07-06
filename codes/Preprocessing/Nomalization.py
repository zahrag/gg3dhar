
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """
import numpy as np


def make_normalization(input_data):

    '''
         This function normalizes 3D joints positions
        '''

    mat = input_data.reshape(15, 3)
    mat[:, 0] /= np.max(mat[:, 0])
    mat[:, 1] /= np.max(mat[:, 1])
    mat[:, 2] /= np.max(mat[:, 2])

    if np.min(mat) < 0:
        mat = mat-np.min(mat)

    output_data = mat.reshape(1, 45)

    return output_data
