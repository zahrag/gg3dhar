
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np


def make_attention(input_data, n_act, dataset):

    '''
        This function applies an attention mechanism as a pre-processing layer based on the joint velocity while acting.
        Attention mechanism is applied in running experiments using MSR-Actio3D datasets only.

        MSR_Action3D_1: The actions of this dataset are all performed using left arm, and the motion velocities of
        4 joints of the left arms were averagely superior compared to all other joints. We applied
        two different attention approaches in joints selection but the results reported in the papers are based on app1.

        MSR_Action3D_2: The actions of this dataset are performed by the joints of arms as well as legs.
        Based on the joints velocity, we made selection of 4 joints with averagely maximum motion velocity while acting.
        Further investigations of joints selection playing bigger role in acting could be done in future studies.

        MSR_Action3D_all: This set is composed of the actions of MSR_Action3D_1 and MSR_Action3D_2.

        '''

    output_data = input_data
    input_data = np.expand_dims(input_data, 1)

    # MSR
    # Body and Head: J7(stomach/center of hips), J4(torso), J3(neck), J20(head)
    # Right Arm: (J3) J1, J8, J10, J12
    # Left Arm:  (J3) J2, J9, J11, J13
    # Right Leg: (J7) J5, J14, J16, J18
    # Left Leg:  (J7) J6, J15, J17, J19

    if dataset == 'MSR_Action3D_1':

        output_data = np.zeros((1, 12))
        app = 1
        if app == 1:
            # Left Arm (J2, J9, J11, J13)
            output_data[0, 0] = input_data[3, 0]
            output_data[0, 1] = input_data[4, 0]
            output_data[0, 2] = input_data[5, 0]
            output_data[0, 3] = input_data[24, 0]
            output_data[0, 4] = input_data[25, 0]
            output_data[0, 5] = input_data[26, 0]
            output_data[0, 6] = input_data[30, 0]
            output_data[0, 7] = input_data[31, 0]
            output_data[0, 8] = input_data[32, 0]
            output_data[0, 9] = input_data[36, 0]
            output_data[0, 10] = input_data[37, 0]
            output_data[0, 11] = input_data[38, 0]

        elif app == 2:
            # Left Arm (J9, J11)
            output_data[0, 0] = input_data[24, 0]
            output_data[0, 1] = input_data[25, 0]
            output_data[0, 2] = input_data[26, 0]
            output_data[0, 3] = input_data[30, 0]
            output_data[0, 4] = input_data[31, 0]
            output_data[0, 5] = input_data[32, 0]
            # Right Arm (J8, J10)
            output_data[0, 6] = input_data[21, 0]
            output_data[0, 7] = input_data[22, 0]
            output_data[0, 8] = input_data[23, 0]
            output_data[0, 9] = input_data[27, 0]
            output_data[0, 10] = input_data[28, 0]
            output_data[0, 11] = input_data[29, 0]

    elif dataset == 'MSR_Action3D_2':
        output_data = np.zeros((1, 12))
        if n_act in [0, 1, 2, 7, 8, 9]:
            output_data[0, 0] = input_data[21, 0]
            output_data[0, 1] = input_data[22, 0]
            output_data[0, 2] = input_data[23, 0]
            output_data[0, 3] = input_data[27, 0]
            output_data[0, 4] = input_data[28, 0]
            output_data[0, 5] = input_data[29, 0]
            output_data[0, 6] = input_data[24, 0]
            output_data[0, 7] = input_data[25, 0]
            output_data[0, 8] = input_data[26, 0]
            output_data[0, 9] = input_data[30, 0]
            output_data[0, 10] = input_data[31, 0]
            output_data[0, 11] = input_data[32, 0]

        if n_act == 3:
            output_data[0, 0] = input_data[57, 0]
            output_data[0, 1] = input_data[58, 0]
            output_data[0, 2] = input_data[59, 0]
            output_data[0, 3] = input_data[6, 0]
            output_data[0, 4] = input_data[7, 0]
            output_data[0, 5] = input_data[8, 0]
            output_data[0, 6] = input_data[9, 0]
            output_data[0, 7] = input_data[10, 0]
            output_data[0, 8] = input_data[11, 0]
            output_data[0, 9] = input_data[18, 0]
            output_data[0, 10] = input_data[19, 0]
            output_data[0, 11] = input_data[20, 0]

        if n_act in [4, 5]:
            output_data[0, 0] = input_data[39, 0]
            output_data[0, 1] = input_data[40, 0]
            output_data[0, 2] = input_data[41, 0]
            output_data[0, 3] = input_data[45, 0]
            output_data[0, 4] = input_data[46, 0]
            output_data[0, 5] = input_data[47, 0]
            output_data[0, 6] = input_data[42, 0]
            output_data[0, 7] = input_data[43, 0]
            output_data[0, 8] = input_data[44, 0]
            output_data[0, 9] = input_data[48, 0]
            output_data[0, 10] = input_data[49, 0]
            output_data[0, 11] = input_data[50, 0]

        if n_act == 6:
            output_data[0, 0] = input_data[27, 0]
            output_data[0, 1] = input_data[28, 0]
            output_data[0, 2] = input_data[29, 0]
            output_data[0, 3] = input_data[30, 0]
            output_data[0, 4] = input_data[31, 0]
            output_data[0, 5] = input_data[32, 0]
            output_data[0, 6] = input_data[45, 0]
            output_data[0, 7] = input_data[46, 0]
            output_data[0, 8] = input_data[47, 0]
            output_data[0, 9] = input_data[48, 0]
            output_data[0, 10] = input_data[49, 0]
            output_data[0, 11] = input_data[50, 0]

    elif dataset == 'MSR_Action3D_all':
        output_data = np.zeros((1, 12))
        if n_act < 10:
            output_data[0, 0] = input_data[3, 0]
            output_data[0, 1] = input_data[4, 0]
            output_data[0, 2] = input_data[5, 0]
            output_data[0, 3] = input_data[24, 0]
            output_data[0, 4] = input_data[25, 0]
            output_data[0, 5] = input_data[26, 0]
            output_data[0, 6] = input_data[30, 0]
            output_data[0, 7] = input_data[31, 0]
            output_data[0, 8] = input_data[32, 0]
            output_data[0, 9] = input_data[36, 0]
            output_data[0, 10] = input_data[37, 0]
            output_data[0, 11] = input_data[38, 0]

        if n_act in [0+10, 1+10, 2+10, 7+10, 8+10, 9+10]:
            output_data[0, 0] = input_data[21, 0]
            output_data[0, 1] = input_data[22, 0]
            output_data[0, 2] = input_data[23, 0]
            output_data[0, 3] = input_data[27, 0]
            output_data[0, 4] = input_data[28, 0]
            output_data[0, 5] = input_data[29, 0]
            output_data[0, 6] = input_data[24, 0]
            output_data[0, 7] = input_data[25, 0]
            output_data[0, 8] = input_data[26, 0]
            output_data[0, 9] = input_data[30, 0]
            output_data[0, 10] = input_data[31, 0]
            output_data[0, 11] = input_data[32, 0]

        if n_act == 3+10:
            output_data[0, 0] = input_data[57, 0]
            output_data[0, 1] = input_data[58, 0]
            output_data[0, 2] = input_data[59, 0]
            output_data[0, 3] = input_data[6, 0]
            output_data[0, 4] = input_data[7, 0]
            output_data[0, 5] = input_data[8, 0]
            output_data[0, 6] = input_data[9, 0]
            output_data[0, 7] = input_data[10, 0]
            output_data[0, 8] = input_data[11, 0]
            output_data[0, 9] = input_data[18, 0]
            output_data[0, 10] = input_data[19, 0]
            output_data[0, 11] = input_data[20, 0]

        if n_act in [4+10, 5+10]:
            output_data[0, 0] = input_data[39, 0]
            output_data[0, 1] = input_data[40, 0]
            output_data[0, 2] = input_data[41, 0]
            output_data[0, 3] = input_data[45, 0]
            output_data[0, 4] = input_data[46, 0]
            output_data[0, 5] = input_data[47, 0]
            output_data[0, 6] = input_data[42, 0]
            output_data[0, 7] = input_data[43, 0]
            output_data[0, 8] = input_data[44, 0]
            output_data[0, 9] = input_data[48, 0]
            output_data[0, 10] = input_data[49, 0]
            output_data[0, 11] = input_data[50, 0]

        if n_act == 6+10:
            output_data[0, 0] = input_data[27, 0]
            output_data[0, 1] = input_data[28, 0]
            output_data[0, 2] = input_data[29, 0]
            output_data[0, 3] = input_data[30, 0]
            output_data[0, 4] = input_data[31, 0]
            output_data[0, 5] = input_data[32, 0]
            output_data[0, 6] = input_data[45, 0]
            output_data[0, 7] = input_data[46, 0]
            output_data[0, 8] = input_data[47, 0]
            output_data[0, 9] = input_data[48, 0]
            output_data[0, 10] = input_data[49, 0]
            output_data[0, 11] = input_data[50, 0]

    return output_data
