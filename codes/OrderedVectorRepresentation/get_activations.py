
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import math
from numpy import linalg as LA


def extract_unique_activation(all_activity_pattern):

    pattern_vector = []
    pat_num = np.size(all_activity_pattern, 0)
    lp = np.zeros((1, pat_num))
    for nseq in range(pat_num):
        mat = all_activity_pattern[nseq]
        winner_x = mat[0, 0]
        winner_y = mat[0, 1]
        mat_n = 1000*np.ones((1, 2))
        for nfr in range(np.size(mat, 0)):
            if mat[nfr, 0] != winner_x or mat[nfr, 1] != winner_y:
                vec = np.array([[winner_x, winner_y]])
                mat_n = np.vstack((mat_n, vec))
                winner_x = mat[nfr, 0]
                winner_y = mat[nfr, 1]

        if mat_n[-1, 0] != winner_x or mat_n[-1, 1] != winner_y:
            vec = np.array([[winner_x, winner_y]])
            mat_n = np.vstack((mat_n, vec))

        pattern_vector.append(mat_n[1:np.size(mat_n, 0), :])
        lp[0, nseq] = np.size(mat_n, 0)-1
        del mat_n

    l_max = np.max(lp)

    return pattern_vector, l_max


def place_new_node(p1, p2, p3, delta):

    if LA.norm(p2-p1) > delta:
        n = (p2-p1)/LA.norm(p2-p1)
        t = delta/math.sqrt(n[0]**2 + n[1]**2)
        Pn = np.array([n[0]*t+p1[0], n[1]*t+p1[1]])
        com = True

    elif LA.norm(p2-p1) < delta:

        delta1 = delta-LA.norm(p2-p1)
        if LA.norm(p3 - p2) == 0:
            print('p1, p2, p3, delta:', p1, p2, p3, delta)

        n = (p3 - p2) / LA.norm(p3 - p2)

        t = delta1/math.sqrt(n[0]**2 + n[1]**2)
        Pn = np.array([n[0] * t + p1[0], n[1] * t + p1[1]])

        com = False

    else:

        Pn = p2
        com = False

    return Pn, com


def ordered_vector_representation(all_activity_pattern):

    all_pmat, l_m = extract_unique_activation(all_activity_pattern)

    pat_num = np.size(all_activity_pattern, 0)
    l_max = int(l_m)
    all_pmat_n = []
    for nseq in range(pat_num):

        pmat = all_pmat[nseq]
        pmat0 = pmat
        l_seq = np.size(pmat, 0)
        if l_seq == 1:
            for nl in range(l_max-1):
                pmat = np.insert(pmat, -1, pmat[0, :], axis=0)

        else:

            L = 0
            for cnt1 in range(l_seq-1):
                L += LA.norm(pmat[cnt1+1, :]-pmat[cnt1, :])

            delta = L/l_max

            ip = 0
            stop = False
            while not stop:

                if ip == np.size(pmat, 0)-2:
                    Pn = 0.5*(pmat[ip, :]+pmat[ip+1, :])
                    pmat = np.insert(pmat, ip+1, Pn, axis=0)

                if np.size(pmat, 0) == l_max:
                    stop = True

                else:
                    Pn, com = place_new_node(pmat[ip, :], pmat[ip+1, :], pmat[ip+2, :], delta)
                    if com:
                        pmat = np.insert(pmat, ip + 1, Pn, axis=0)
                    else:
                        pmat = np.delete(pmat, ip + 1, axis=0)
                        pmat = np.insert(pmat, ip + 1, Pn, axis=0)

                    ip += 1

        all_pmat_n.append(pmat.flatten())

        del pmat
        del pmat0

    return all_pmat_n, l_max


def get_snn_activations(all_snn_activity, data_index, data_class, nbr_of_class, max_seed):

    activity_map_seq = np.zeros((max_seed, nbr_of_class))
    result_max = 0.0
    result_mean = 0.0
    all_pred_max = []
    all_pred_mean = []
    all_pred_ave = []
    all_true = []
    for cnt_seq in range(len(data_index)):
        for cnt_models in range(max_seed):
            activity_map_seq[cnt_models, :] = all_snn_activity[cnt_models][cnt_seq]

        # True Action
        a_true = data_class[int(data_index[cnt_seq])][2]
        all_true.append(a_true)

        # Approach1:
        a_pred = np.argmax(activity_map_seq, axis=1)
        pred_seq = a_pred == a_true
        all_pred_ave.append(np.sum(pred_seq)/max_seed)

        # Approach2 (Max-Operator): Predicted Action
        a_pred_max = np.argmax(np.max(activity_map_seq, axis=0))
        if a_pred_max == a_true:
            result_max += 1
        all_pred_max.append(a_pred_max)

        # Approach3 (Mean-Operator): Predicted Action
        a_pred_mean = np.argmax(np.mean(activity_map_seq, axis=0))
        if a_pred_mean == a_true:
            result_mean += 1
        all_pred_mean.append(a_pred_mean)

    result_max_operator = 100 * result_max / len(data_index)
    result_mean_operator = 100 * result_mean / len(data_index)
    result = 100 * np.sum(all_pred_ave) / len(data_index)

    return result, result_max_operator, result_mean_operator, all_true, all_pred_ave, all_pred_max, all_pred_mean


