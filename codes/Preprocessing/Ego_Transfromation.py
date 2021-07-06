
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from numpy import linalg as LA


def make_egoCenteredCoordinateT(input_vec, dataset):

    '''
        This function maps all 3D joints position into a new right-hand coordinate system centered at the joint stomach
        '''

    input_vec = np.expand_dims(input_vec, 0)

    if dataset == 'MSR_Action3D_2' or dataset == 'MSR_Action3D_1':
        p_1 = np.array([[input_vec[0, 12], input_vec[0, 13], input_vec[0, 14]]])  # Right Hip
        p_2 = np.array([[input_vec[0, 15], input_vec[0, 16], input_vec[0, 17]]])  # Left Hip
        p_3 = np.array([[input_vec[0, 18], input_vec[0, 19], input_vec[0, 20]]])  # Stomach
        nbr_joint = 20

    elif dataset == 'UTKinect':
        p_1 = np.array([[input_vec[0, 36], input_vec[0, 37], input_vec[0, 38]]])  # Right Hip
        p_2 = np.array([[input_vec[0, 48], input_vec[0, 49], input_vec[0, 50]]])  # Left Hip
        p_3 = np.array([[input_vec[0, 0], input_vec[0, 1], input_vec[0, 2]]])  # Stomach
        nbr_joint = 20

    else:  # Florence
        p_1 = np.array([[input_vec[0, 27], input_vec[0, 28], input_vec[0, 29]]])  # Right Hip
        p_2 = np.array([[input_vec[0, 36], input_vec[0, 37], input_vec[0, 38]]])  # Left Hip
        p_3 = np.array([[input_vec[0, 6], input_vec[0, 7], input_vec[0, 8]]])  # Stomach
        nbr_joint = 15

    d_p12 = p_2 - p_1

    d_p12_n = d_p12 / LA.norm(d_p12)

    d_p31 = p_1 - p_3

    num = np.sum(d_p12 * d_p31)
    den = np.sum(d_p12 * d_p12_n)
    t = -num / den

    # p4: stomach (p3) projection on line connecting p1 to p2
    p_4 = d_p12_n * t + p_1

    # Y-axis
    d_p42 = p_2 - p_4

    # Z-axis
    d_p43 = p_3 - p_4

    # X-axis
    c_p = np.zeros((1, 3))
    c_p[0, 0] = d_p42[0, 1] * d_p43[0, 2] - d_p42[0, 2] * d_p43[0, 1]
    c_p[0, 1] = d_p42[0, 2] * d_p43[0, 0] - d_p42[0, 0] * d_p43[0, 2]
    c_p[0, 2] = d_p42[0, 0] * d_p43[0, 1] - d_p42[0, 1] * d_p43[0, 0]

    # normalized axises
    yB = d_p42 / LA.norm(d_p42)
    zB = d_p43 / LA.norm(d_p43)
    xB = c_p / LA.norm(c_p)

    xA = np.array([[1, 0, 0]])
    yA = np.array([[0, 1, 0]])
    zA = np.array([[0, 0, 1]])

    A_R_B = np.zeros((4, 4))
    A_R_B[0, 0] = np.sum(xB * xA)
    A_R_B[0, 1] = np.sum(yB * xA)
    A_R_B[0, 2] = np.sum(zB * xA)
    A_R_B[0, 3] = p_3[0, 0]

    A_R_B[1, 0] = np.sum(xB * yA)
    A_R_B[1, 1] = np.sum(yB * yA)
    A_R_B[1, 2] = np.sum(zB * yA)
    A_R_B[1, 3] = p_3[0, 1]

    A_R_B[2, 0] = np.sum(xB * zA)
    A_R_B[2, 1] = np.sum(yB * zA)
    A_R_B[2, 2] = np.sum(zB * zA)
    A_R_B[2, 3] = p_3[0, 2]

    A_R_B[3, 0] = 0
    A_R_B[3, 1] = 0
    A_R_B[3, 2] = 0
    A_R_B[3, 3] = 1

    if LA.det(A_R_B) != 0:
        iT = LA.inv(A_R_B)

        for id in range(nbr_joint):
            p_id = np.array([[input_vec[0, 3 * id + 0]], [input_vec[0, 3 * id + 1]], [input_vec[0, 3 * id + 2]], [1]])
            p_id_n = iT @ p_id
            input_vec[0, 3 * id + 0] = p_id_n[0, 0]
            input_vec[0, 3 * id + 1] = p_id_n[1, 0]
            input_vec[0, 3 * id + 2] = p_id_n[2, 0]

    return input_vec
