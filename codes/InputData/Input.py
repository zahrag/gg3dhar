
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from Preprocessing.Nomalization import make_normalization
from Preprocessing.Attection import make_attention
from Preprocessing.Ego_Transfromation import make_egoCenteredCoordinateT
from Preprocessing.Dynamics import get_dynamics
from InputData.read_files import read_MSR, read_Florence, read_UTKinect


class DATA:
    """
       This class generates preprocessed input data.

       """

    def __init__(self, input_dim=60, mainpath=None, dataset=None):

        self.mainpath = mainpath
        self.Dataset = dataset
        self.input_dim = input_dim

        self.actionSet = []
        if self.Dataset == 'MSR_Action3D_1':
            self.actionSet = ['High-Wave',
                              'Front-Wave',
                              'Using-Hammer',
                              'Hand-Catch',
                              'Forward-Punch',
                              'High-Throw',
                              'Draw-Xsign',
                              'Draw-TickSign',
                              'Draw-Circle',
                              'Tennis-Swing',
                              ]

            self.prepro_attention = True
            self.prepro_ego = True
            self.prepro_norm = False
            self.prepro_scaling = True
            self.prepro_dyn = True

        elif self.Dataset == 'MSR_Action3D_2':
            self.actionSet = ['Hand_Clap',
                              'Two-Hand-Wave',
                              'Side-Boxing',
                              'Forward-Bend',
                              'Forward-Kick',
                              'Side-Kick',
                              'Still-Jogging',
                              'Tennis-Serve',
                              'Golf-Swing',
                              'PickUp-Throw',
                              ]
            self.prepro_attention = True
            self.prepro_ego = True
            self.prepro_norm = False
            self.prepro_scaling = False
            self.prepro_dyn = True

        elif self.Dataset == 'MSR_Action3D_all':
            self.actionSet = ['High-Wave',
                              'Front-Wave',
                              'Using-Hammer',
                              'Hand-Catch',
                              'Forward-Punch',
                              'High-Throw',
                              'Draw-Xsign',
                              'Draw-TickSign',
                              'Draw-Circle',
                              'Tennis-Swing',
                              'Hand_Clap',
                              'Two-Hand-Wave',
                              'Side-Boxing',
                              'Forward-Bend',
                              'Forward-Kick',
                              'Side-Kick',
                              'Still-Jogging',
                              'Tennis-Serve',
                              'Golf-Swing',
                              'PickUp-Throw',
                              ]
            self.prepro_attention = True
            self.prepro_ego = True
            self.prepro_norm = False
            self.prepro_scaling = True
            self.prepro_dyn = True

        elif self.Dataset == 'Florence':
            self.actionSet = ['High-Wave',
                              'Drink-Bottle',
                              'Answer-Cellphone',
                              'Hand-Clap',
                              'Tight-Lace',
                              'Sit-Down',
                              'Stand-Up',
                              'Read-watch',
                              'Make-Bow',
                              ]
            self.prepro_attention = False
            self.prepro_ego = True
            self.prepro_norm = False
            self.prepro_scaling = False
            self.prepro_dyn = True

        elif self.Dataset == 'UTKinect':
            self.actionSet = ['Walking',
                              'Sit-Down',
                              'Stand-Up',
                              'Pick-Up',
                              'Carrying',
                              'Throwing',
                              'Pushing',
                              'Pulling',
                              'Wave-Hand',
                              'Clap-Hand',
                              ]
            self.prepro_attention = False
            self.prepro_ego = True
            self.prepro_norm = False
            self.prepro_scaling = True
            self.prepro_dyn = True

        self.pos_all = []
        self.pos_all_n = []
        self.vel_all = []
        self.acc_all = []
        self.class_all = []

    def read_data(self):

        """
           This function reads data from files.

           """

        if self.Dataset == 'MSR_Action3D_1':
            self.pos_all, self.class_all = read_MSR(self.mainpath, self.pos_all, self.class_all, set=1)

        elif self.Dataset == 'MSR_Action3D_2':
            self.pos_all, self.class_all = read_MSR(self.mainpath, self.pos_all, self.class_all, set=2)

        elif self.Dataset == 'MSR_Action3D_all':
            self.pos_all, self.class_all = read_MSR(self.mainpath, self.pos_all, self.class_all, set=1)
            self.pos_all, self.class_all = read_MSR(self.mainpath, self.pos_all, self.class_all, set=2)
            l_act = []
            [l_act.append(np.array([0, 0, 0, 0])) for k in range(276)]
            [l_act.append(np.array([276, 0, 10, 0])) for k in range(276, len(self.class_all))]
            self.class_all = [self.class_all[k] + l_act[k] for k in range(len(self.class_all))]

        elif self.Dataset == 'Florence':
            self.pos_all, self.class_all = read_Florence(self.mainpath, self.pos_all, self.class_all)

        elif self.Dataset == 'UTKinect':
            self.pos_all, self.class_all = read_UTKinect(self.mainpath, self.pos_all, self.class_all)

    def make_preprocessing(self):

        """
           This function runs pre-processing module consists of:
           (1) Normalization
           (2) Ego-Centered Coordinate Transformation
           (3) Scaling Transformation
           (4) Attention Mechanisms
           (5) Dynamics Extraction

           """

        for nseq in range(len(self.pos_all)):  # sequence

            data_seq = self.pos_all[nseq]
            class_seq = self.class_all[nseq]
            n_act = class_seq[2]

            data_seq_n = np.zeros((np.size(data_seq, 0), self.input_dim))
            for nfr in range(np.size(data_seq, 0)):  # frame

                if self.prepro_norm:
                    data_seq[nfr, :] = make_normalization(data_seq[nfr, :])

                if self.prepro_ego:
                    data_seq[nfr, :] = make_egoCenteredCoordinateT(data_seq[nfr, :], self.Dataset)

                if self.prepro_attention:
                    vec = make_attention(data_seq[nfr, :], n_act, self.Dataset)
                    data_seq_n[nfr, :] = vec[0, :]

            if self.prepro_attention:
                self.pos_all_n.append(data_seq_n)
            else:
                self.pos_all_n.append(data_seq)

        if self.prepro_dyn:
            self.vel_all, self.acc_all = get_dynamics(self.pos_all_n)

    def get_input(self):

        # Read data from files
        self.read_data()

        # Do the pre-processing
        self.make_preprocessing()












