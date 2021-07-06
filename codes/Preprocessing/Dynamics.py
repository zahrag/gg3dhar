
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np


def get_dynamics(data):

    '''
        This function extracts first and second orders of dynamic representing velocity and acceleration of joints.
       '''

    vel = []
    acc = []

    for nseq in range(len(data)):  # sequence counter

        data_seq = data[nseq]

        vel_seq = np.zeros((np.size(data_seq, 0), np.size(data_seq, 1)))
        acc_seq = np.zeros((np.size(data_seq, 0), np.size(data_seq, 1)))

        pos_0 = np.zeros((1, np.size(data_seq, 1)))
        pos_1 = np.zeros((1, np.size(data_seq, 1)))
        pos_2 = np.zeros((1, np.size(data_seq, 1)))

        for nfr in range(np.size(data_seq, 0)):  # frame counter

            pos_2 = pos_1
            pos_1 = pos_0
            pos_0 = data_seq[nfr, :]

            if nfr > 0:
                vel_seq[nfr, :] = pos_0-pos_1

            if nfr > 1:
                vel_0 = pos_0 - pos_1
                vel_1 = pos_1 - pos_2
                acc_seq[nfr, :] = vel_0 - vel_1

        vel.append(vel_seq)
        acc.append(acc_seq)

    return vel, acc




