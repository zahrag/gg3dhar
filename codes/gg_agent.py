
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
from GG import GG
from SNN import SNN


class ggagent_phase_I:

    def __init__(self, learning, input_size, input_number, sigma, gamma, lambda_, softmax_exponent, max_epoch,
                 dyn_as_input):

        self.net_1 = GG(learning=learning,
                        inputsize=input_size,
                        nbr_input=input_number,
                        sigma=sigma,
                        gamma=gamma,
                        lambda_=lambda_,
                        softmax_exponent=softmax_exponent,
                        max_epoch=max_epoch)

        self.dyn_as_input = dyn_as_input

    def run(self, data, data_d1, data_d2, data_index, learning=None):

        self.net_1.learning = learning
        all_activity_pattern = []
        iteration = 0
        epoch = 0
        run = True
        while run:

            epoch += 1
            if self.net_1.ft_phase:
                self.net_1.ft_epoch += 1

            # Random selection
            rseq = np.random.permutation(len(data_index))
            all_activity_pattern = []
            for nseq in range(len(data_index)):  # Sequences

                if learning is False:
                    ind_seq = int(data_index[nseq])
                else:
                    ind_seq = int(data_index[rseq[nseq]])

                data_seq = data[ind_seq]
                if self.dyn_as_input == 1:
                    data_seq = np.concatenate((data_seq, data_d1[ind_seq]), axis=1)

                elif self.dyn_as_input == 2:
                    data_seq = np.concatenate((data_seq, data_d1[ind_seq]), axis=1)
                    data_seq = np.concatenate((data_seq, data_d2[ind_seq]), axis=1)

                activity_pattern = np.zeros((np.size(data_seq, 0), 2))
                for nfr in range(np.size(data_seq, 0)):  # Frames per sequence
                    iteration += 1

                    activity, winner = self.net_1.growing_grid(data_seq[nfr, :])
                    activity_pattern[nfr, 0] = winner[0]
                    activity_pattern[nfr, 1] = winner[1]

                all_activity_pattern.append(activity_pattern)

            if learning:
                print("", end='\r')
                print("Phase:{}  \t Epoch:{} \t Row:{} \t Column:{}".format(1, epoch, np.size(self.net_1.weights, 0),
                                                                            np.size(self.net_1.weights, 1)), end="",
                      flush=True)

            if epoch == self.net_1.max_epoch or learning is False:
                run = False

        return all_activity_pattern


class ggagent_phase_II:

    def __init__(self, learning, input_size, input_number, sigma, gamma, lambda_, softmax_exponent, max_epoch,
                 class_number):

        self.net_2 = GG(learning=learning,
                        inputsize=input_size,
                        nbr_input=input_number,
                        sigma=sigma,
                        gamma=gamma,
                        lambda_=lambda_,
                        softmax_exponent=softmax_exponent,
                        max_epoch=max_epoch)

        self.net_3 = SNN(learning=learning,
                         outputsize_x=class_number,
                         outputsize_y=1,
                         inputsize=1)

    def run(self, data, data_index, data_class_info, learning=None):

        self.net_2.learning = learning
        self.net_3.learning = learning
        # Performance results
        result_per_class = np.zeros((1, self.net_3.outputsize_x))
        snn_activity = []
        snn_activity_map = []

        epoch = 0
        run = True
        while run:

            epoch += 1
            if self.net_2.ft_phase is True:
                self.net_2.ft_epoch += 1

            rseq = np.random.permutation(len(data_index))
            for nseq in range(len(data_index)):

                if learning is False:
                    ind_seq = int(data_index[nseq])
                else:
                    ind_seq = int(data_index[rseq[nseq]])

                class_seq = data_class_info[ind_seq]
                # running second-layer GG
                activity, winner = self.net_2.growing_grid(data[ind_seq])

                # Re-Initialize SNN Based on the grown grid
                if learning and self.net_2.ft_phase and self.net_2.ft_epoch == 0:
                    self.net_3.inputsize = self.net_2.outputsize_y * self.net_2.outputsize_x
                    self.net_3.weights = np.random.rand(self.net_3.outputsize_x, self.net_3.outputsize_y, self.net_3.inputsize)

                # running third-layer supervised neural network
                if self.net_2.ft_phase:
                    snn_activity, snn_result = self.net_3.run_SNN(activity.flatten(), int(class_seq[2]))
                    result_per_class[0, int(class_seq[2])] += snn_result

                # get third-layer activation maps for Test sequences
                if learning is False:
                    snn_activity_map.append(snn_activity.T)

            if learning:
                print("", end='\r')
                print("Phase:{}  \t Epoch:{} \t Row:{} \t Column:{}".format(2, epoch, np.size(self.net_2.weights, 0),
                                                                            np.size(self.net_2.weights, 1)), end="",
                      flush=True)

            if epoch == self.net_2.max_epoch or learning is False:
                run = False

        return result_per_class, snn_activity_map

 
