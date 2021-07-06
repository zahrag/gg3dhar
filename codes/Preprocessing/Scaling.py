
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from numpy import linalg as LA


def scaling(input_data):

    '''
        This functions applies an scaling mechanism to re-scale links connecting the two consecutive joints based on an
        standard measure set by user.
        :param input_data:
        :return: Re-scaled skeleton
        '''

    # MSR
    # Body and Head: J7, J4, J3, J20
    # Right Arm: J3, J1, J8, J10, J12
    # Left Arm:  J3, J2, J9, J11, J13
    # Right Leg: J7, J5, J14, J16, J18
    # Left Leg:  J7, J6, J15, J17, J19

    out = np.zeros(np.size(input_data))
    st_lint = 42.0951
    out[18] = input_data[18]
    out[19] = input_data[19]
    out[20] = input_data[20]
    ind = [18, 9]
    out = get_scaled_position(input_data, ind, st_lint-0.35*st_lint)

    st_lint = 53.6656
    ind = [9, 6]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 58.2087
    ind = [6, 57]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 40.9420
    ind = [6, 0]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 57.2800
    ind = [0, 21]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 62.4197
    ind = [21, 27]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 14.2215
    ind = [27, 33]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 42.2374
    ind = [6, 3]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 57.2735
    ind = [3, 24]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 62.7077
    ind = [24, 30]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 14.0089
    ind = [30, 36]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 35.1888
    ind = [18, 12]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 122.3575
    ind = [12, 39]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 82.8312
    ind = [39, 45]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 30.0832
    ind = [45, 51]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 32.4962
    ind = [18, 15]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 120.3536
    ind = [15, 42]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 86.1409
    ind = [42, 48]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    st_lint = 27.7308
    ind = [48, 54]
    out = get_scaled_position(input_data, ind, st_lint - 0.35 * st_lint)

    return out


def get_scaled_position(input_data, ind, L):

    n_joints = int(np.size(input_data)/3)
    mat = input_data.reshape(n_joints, 3)
    N = (mat[ind[1], :] - mat[ind[0], :]) / LA.norm(mat[ind[1], :] - mat[ind[0], :])

    t1 = L / LA.norm(N)
    t2 = -L / LA.norm(N)

    J0 = mat[ind[0], :]
    J1 = N*t1 + J0

    N1 = (J1-J0) / LA.norm(J1-J0)

    cos_theta = np.sum(N*N1)/(LA.norm(N)*LA.norm(N1))

    if cos_theta < 0:
        J1 = N * t2 + J0

    mat[ind[1], :] = J1
    out = mat.flatten()

    return out