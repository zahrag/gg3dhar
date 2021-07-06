
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import itertools


class RS:
    """
        This class generates random selection of the dataset

        """
    def __init__(self, nbr_of_class=10, nbr_of_folds=10, ratio=0.25):

        self.nbr_fold = nbr_of_folds
        self.nbr_class = nbr_of_class
        self.ratio = ratio
        self.tr_set = []
        self.val_set = []
        self.ts_set = []
        self.tr_val_set = []
        self.n_trseq_per_act = np.zeros((1, nbr_of_class))
        self.n_valseq_per_act = np.zeros((1, nbr_of_class))
        self.n_tsseq_per_act = np.zeros((1, nbr_of_class))
        self.n_trvalseq_per_act = np.zeros((1, nbr_of_class))
        self.fold = []
        self.nfr_tr = 0

    def random_test_selector(self, class_all):
        """
            This function creates the randomly selected test data set based on a ratio predefined by the user.

            """
        sum_seq_per_act = np.zeros((1, self.nbr_class))
        seq_per = np.zeros((self.nbr_class, 50))
        for nseq in range(len(class_all)):
            info_seq = class_all[nseq]
            cnt_act = int(info_seq[2])
            cnt_seq = int(sum_seq_per_act[0, cnt_act])
            seq_per[cnt_act, cnt_seq] = nseq
            sum_seq_per_act[0, cnt_act] += 1

        seq_per_act = []
        for na in range(self.nbr_class):
            cnt_seq = int(sum_seq_per_act[0, na])
            vec = seq_per[na, 0:cnt_seq]
            seq_per_act.append(vec)

        for na in range(self.nbr_class):
            ract = np.random.permutation(int(sum_seq_per_act[0, na]))
            info_seq = seq_per_act[na]
            n_ts = round(self.ratio*len(info_seq))
            for nseq in range(len(info_seq)):
                if nseq < n_ts:
                    self.ts_set.append(info_seq[ract[nseq]])
                    self.n_tsseq_per_act[0, na] += 1
                else:
                    self.tr_val_set.append(info_seq[ract[nseq]])
                    self.n_trvalseq_per_act[0, na] += 1

    def n_fold_selector(self, class_all):
        """
            Once in the start of running HAR architecture,
            this function randomly divides the whole dataset into one test set and n-folds for training.
            The number of folds is pre-defined by the user.

            """
        self.random_test_selector(class_all)

        r_seq = np.random.permutation(int(len(self.tr_val_set)))  # Randomization

        n_val = int(np.round((1/self.nbr_fold)*len(self.tr_val_set)))

        for cnt1 in range(self.nbr_fold):
            vec = []
            for cnt2 in range(n_val*cnt1, n_val*(cnt1+1)):
                if cnt2 < len(self.tr_val_set):
                    cnt_seq = self.tr_val_set[r_seq[cnt2]]
                    vec.append(cnt_seq)
            self.fold.append(vec)

        return

    def random_selector_nfold(self, input_all, class_all, seed):
        """
            For each running seeds,
            this function randomly creates training and validation sets based on n-fold cross validation approach.

            """
        # fold number
        if seed > self.nbr_fold:
            nfold = int(seed % len(self.fold))
        else:
            nfold = seed

        # validation set
        self.val_set = self.fold[nfold]
        self.n_valseq_per_act = np.zeros((1, self.nbr_class))
        for nseq in range(len(self.val_set)):
            info = class_all[int(self.val_set[nseq])]
            self.n_valseq_per_act[0, int(info[2])] += 1

        # training set
        tr_set = []
        for nf in range(len(self.fold)):
            if nf != nfold:
                tr_set.append(self.fold[nf])
        self.tr_set = list(itertools.chain.from_iterable(tr_set))

        # number of sequence per action ¥ frames
        self.nfr_tr = 0
        self.n_trseq_per_act = np.zeros((1, self.nbr_class))
        for nseq in range(len(self.tr_set)):
            data_mat = input_all[int(self.tr_set[nseq])]
            self.nfr_tr += np.size(data_mat, 0)
            info = class_all[int(self.tr_set[nseq])]
            self.n_trseq_per_act[0, int(info[2])] += 1

    def random_selector_basic(self, input_all, class_all):
        """
            For each running seeds,
            this function randomly creates training and validation sets,
            without applying  n-fold cross validation approach.

            """
        self.random_test_selector(class_all)

        self.tr_set = self.tr_val_set
        self.val_set = self.tr_val_set

        self.n_trseq_per_act = self.n_trvalseq_per_act
        self.n_valseq_per_act = self.n_trvalseq_per_act

        # self.tr_set = list(itertools.chain.from_iterable(tr_set))
        self.nfr_tr = 0
        for nseq in range(len(self.tr_set)):
            data_mat = input_all[int(self.tr_set[nseq])]
            self.nfr_tr += np.size(data_mat, 0)

