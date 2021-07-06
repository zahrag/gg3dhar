
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_activation_pattern(pattern, pattern_n, net, rs, datainfo, seed, path, illus=None, save_figs=None):
    """
       This function plots the action pattern vectors.
       Args:
            pattern: Original action pattern vectors.
            pattern_n: Generated action pattern vectors by applying superimposition layer.
            net: Network designed and built for phase_I.
            rs: Random selector module.
            datainfo: Input dataset information.
            seed: Current running seed.
            path: Path to save patterns.
            illus: Flag to permit illustration.
            save_figs: Flag to permit saving patterns.

        """

    if illus:
        patt_path = path + '/' + 'model{}/'.format(seed + 1)
        if not os.path.exists(patt_path):
            os.makedirs(patt_path)

        l_x = net.outputsize_x
        l_y = net.outputsize_y
        L_max = int(np.size(pattern_n, 1) / 2)

        # Train & Validation sets
        make_plots(pattern, pattern_n, datainfo.class_all, L_max, l_x, l_y, rs.nbr_class, rs.tr_val_set, patt_path,
                   train=True, save_figs=save_figs)

        # Test set
        make_plots(pattern, pattern_n, datainfo.class_all, L_max, l_x, l_y, rs.nbr_class, rs.ts_set, patt_path,
                   train=False, save_figs=save_figs)


def make_plots(pattern, pattern_n, class_info, L_max, l_x, l_y, nbr_of_class, set_index, path,
               train=None, save_figs=None):

    vec_pat = []
    vec_pat_n = []
    for i in range(nbr_of_class):
        vec_pat.append([])
        vec_pat_n.append([])
    for nseq in range(len(set_index)):
        seq_index = int(set_index[nseq])
        class_label = int(class_info[seq_index][2])
        vec_pat[class_label].append(pattern[seq_index])
        pattern_n[seq_index].resize(L_max, 2)
        vec_pat_n[class_label].append(pattern_n[seq_index])

    for na in range(nbr_of_class):
        if len(vec_pat[na]) % 2 == 0:
            n_cols = int(len(vec_pat[na]) / 2)
        else:
            n_cols = int((len(vec_pat[na]) + 1) / 2)
        fig, axs = plt.subplots(2, n_cols, figsize=(15, 6), facecolor='w', edgecolor='k')
        # axs.set_title('Action:{}'.format(na))

        for ns in range(len(vec_pat[na])):
            vec = vec_pat[na][ns]
            vec_n = vec_pat_n[na][ns]
            if ns < n_cols:
                axs[0, ns].plot(vec[:, 0], vec[:, 1], 'k--',
                                vec_n[:, 0], vec_n[:, 1], 'ko')
                axs[0, ns].set_xlim([0, l_x])
                axs[0, ns].set_ylim([0, l_y])

            else:
                ns_ = int(ns % n_cols)
                axs[1, ns_].plot(vec[:, 0], vec[:, 1], 'k--',
                                 vec_n[:, 0], vec_n[:, 1], 'ko')
                axs[1, ns_].set_xlim([0, l_x])
                axs[1, ns_].set_ylim([0, l_y])

        if save_figs:
            if train:
                plt.savefig('{}Train_Action{}.pdf'.format(path, na), bbox_inches='tight')
            else:
                plt.savefig('{}Test_Action{}.pdf'.format(path, na), bbox_inches='tight')

    plt.show()









