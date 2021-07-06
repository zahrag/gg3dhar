
"""
    Author: Zahra Gharaee.
    This code is written for the Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import gg_agent
from OrderedVectorRepresentation.get_activations import ordered_vector_representation


class Architecture:
    """
        This class builds action recognition architecture in two phases in OFFLINE mode
        Phase_I: consists of one network, SOM/GG/ASOM coupled with the Superimposition module
        Phase_II: consists of two networks, a SOM/GG/ASOM coupled with a SNN

        """

    def __init__(self, input_dim=1, output_dim=1, random_selector=None, Dyn=0):

        # ****************************************************** Initialize Phase I:
        self.phase_I = gg_agent.ggagent_phase_I(learning=True,
                                                input_size=input_dim*(Dyn+1),
                                                input_number=random_selector.nfr_tr,
                                                sigma=10**6,
                                                gamma=30**2,
                                                lambda_=round((random_selector.nfr_tr - 2) / 3),
                                                softmax_exponent=10,
                                                max_epoch=150,
                                                dyn_as_input=Dyn)

        # ****************************************************** Initialize Phase II:
        self.phase_II = gg_agent.ggagent_phase_II(learning=True,
                                                  input_size=1,
                                                  input_number=len(random_selector.tr_set),
                                                  sigma=10**3,
                                                  gamma=40**2,
                                                  lambda_=round((len(random_selector.tr_set) - 2) / 3),
                                                  softmax_exponent=10,
                                                  max_epoch=300,
                                                  class_number=output_dim)

        self.original_patterns = []
        self.new_patterns = []

    def train(self, data, rs):
        """
            This function is to train the architecture in phase_I and phase_II consecutively.
            Trained architecture of phase_I is used to create input of phase_II though superimposition module.

                Args:
                    data: training data sequences and labels
                    rs: random selector object

                Returns:
                    patterns: Original pattern vectors of the whole dataset
                    new_patterns: Ordered vector represented patterns of the whole dataset
                """

        # Training Phase_I:
        _ = self.phase_I.run(data.pos_all_n, data.vel_all, data.acc_all, rs.tr_set, learning=True)

        # Original patterns extraction for the whole dataset (train, validation & test sets)
        self.original_patterns = self.phase_I.run(data.pos_all_n,
                                                  data.vel_all,
                                                  data.acc_all,
                                                  np.vstack(data.class_all)[:, 0],
                                                  learning=False)

        # Ordered Vector Representation
        self.new_patterns, L_max = ordered_vector_representation(self.original_patterns)

        # ----------- Phase_II:
        # Weights Initialization based on new patterns length (L_max)
        self.phase_II.net_2.inputsize = 2*L_max
        self.phase_II.net_2.weights = np.random.rand(self.phase_II.net_2.outputsize_x,
                                                     self.phase_II.net_2.outputsize_y,
                                                     2*L_max)

        # Training Phase_II:
        _ = self.phase_II.run(self.new_patterns, rs.tr_set, data.class_all, learning=True)

        return self.original_patterns, self.new_patterns

    def test(self, patterns, data, rs_set):
        """
           This function runs phase_II in test mode. Since the result of running phase_I is
           represented by action pattern vectors received as the input, running phase_I is test mode
           is not required anymore!

               Args:
                   patterns: ordered vector represented patterns
                   data: input data information
                   rs_set: random selector set corresponds to train, validation or test sets

               Returns:
                   result_perc: recognition result percentage of the data sequences
                   snn_map: SNN activity map for the corresponding data sequences
               """

        result_per_class, snn_map = self.phase_II.run(patterns, rs_set, data.class_all, learning=False)
        result_perc = 100 * np.sum(result_per_class) / float(len(rs_set))

        return result_perc, snn_map

    def print_results_per_seed(self, seed, tr_result, val_result):

        # Printing performance results
        print('\n\n\n Seed:', seed,
              '\n HAR Train-Accuracy:', tr_result,
              '\n HAR Validation-Accuracy:', val_result,
              '\n\n')











