
"""
    Author: Zahra Gharaee.
    This code is written for the 3D-Human-Action-Recognition Project, started March 14 2014.
    """

import numpy as np
import os
from operator import itemgetter

import time
from InputData.Input import DATA
from run_architecture_offline import Architecture
from Save_Load_Data.save_load_data import save_models, load_model
from InputData.RandomSelection import RS
from Illustration.make_illusteration import plot_activation_pattern

if __name__ == '__main__':

    # ---------------------------- Initial Settings ----------------------------
    # Datasets setting
    data_dir = '../DataSets'
    Datasets = ['MSR_Action3D_1', 'MSR_Action3D_2', 'MSR_Action3D_all', 'Florence', 'UTKinect']
    Dataset = Datasets[1]

    # Stage setting
    Stages = ['Training', 'Testing']
    Stage = Stages[0]

    # Data split setting
    make_data_split = True

    # Position dimension setting
    # It depends on the number of joints selected by attention mechanism
    # pose_dim = n_selected_joints x 3(cartesian coordinate: x, y, z)
    pose_dim = 3
    if Dataset == 'MSR_Action3D_1' or Dataset == 'MSR_Action3D_2' or Dataset == 'MSR_Action3D_all':
        pose_dim = 12
    elif Dataset == 'Florence':
        pose_dim = 45
    elif Dataset == 'UTKinect':
        pose_dim = 60

    # Applying orders of dynamics of the joints:
    # Position (0), Position & Velocity (1), Position & Velocity & Acceleration (2)
    Dynamics = [0, 1, 2]
    Dynamic = Dynamics[0]

    # Folds number and test ratio setting
    # We apply 10-fold cross validation and 25% test data selection for input data split
    fold_number = 10
    test_ratio = 0.25

    # Network setting: Growing Grid Neural Network
    net_1 = 'GG'
    net_2 = 'GG'

    # Mode setting
    Mode = 'Offline'

    # Path settings
    log_dir = '../trained_models' + '/' + Dataset
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    patt_dir = log_dir + '/patterns'
    if not os.path.exists(patt_dir):
        os.makedirs(patt_dir)

    # ---------------------------- Make Data Split ----------------------------
    if make_data_split:

        # Read input & do pre-processing
        datainfo = DATA(input_dim=pose_dim, mainpath=data_dir, dataset=Dataset)
        datainfo.get_input()

        # Data split based on 10-fold cross validation
        # Random test selection & folds splits: one random test set for all models
        rs = RS(nbr_of_class=len(datainfo.actionSet), nbr_of_folds=fold_number, ratio=test_ratio)
        rs.n_fold_selector(datainfo.class_all)

        # Set the models' dictionary
        InputData = {
                    'Dataset': Dataset,
                    'Data': datainfo,
                    'Fold_Selector': rs,
                    }

        # Save data split
        datasplit_to_save = log_dir + '/' + 'input_split_{}.pickle'.format(Dataset)
        save_models(datasplit_to_save, InputData, save=True)

    else:
        # Load data split:
        datasplit_to_load = log_dir + '/' + 'input_split_{}.pickle'.format(Dataset)
        InputData = load_model(datasplit_to_load)
        rs = InputData['Fold_Selector']
        datainfo = InputData['Data']

    # ---------------------------- Training the architecture ----------------------------
    if Stage is 'Training':

        # Start of training
        start_time = time.asctime(time.localtime(time.time()))
        print('start time:', start_time)

        arcs = []
        fs = []
        tr_results_all = []
        val_results_all = []
        ts_results_all = []
        for seed in range(rs.nbr_fold):

            print("", end='\r')
            print("Seed:{}".format(seed+1), end="", flush=False)

            # Random selection of train and validation data set per seed
            rs.random_selector_nfold(datainfo.pos_all_n, datainfo.class_all, seed)

            # Initialize Architecture
            arc = Architecture(input_dim=np.size(datainfo.pos_all_n[0], 1),
                               output_dim=len(datainfo.actionSet),
                               random_selector=rs,
                               Dyn=Dynamic)

            # Training model
            original_patterns, new_patterns = arc.train(datainfo, rs)

            # Illustration
            plot_activation_pattern(original_patterns, new_patterns, arc.phase_I.net_1, rs, datainfo,
                                    seed, patt_dir, illus=False, save_figs=False)

            # Testing model
            tr_result, tr_snn_map = arc.test(new_patterns, datainfo, rs.tr_set)
            val_result, val_snn_map = arc.test(new_patterns, datainfo, rs.val_set)

            # Print results per seed
            arc.print_results_per_seed(seed, tr_result, val_result)

            # Collecting results and models
            tr_results_all.append(tr_result)
            val_results_all.append(val_result)
            arcs.append(arc)
            fs.append(rs)

        # End of training
        end_time = time.asctime(time.localtime(time.time()))
        print('\n end time:', end_time)

        # Set the models' dictionary
        ModelSet = {
                'Dataset': Dataset,
                'Mode': Mode,
                'Start_Time': start_time,
                'End_Time': end_time,
                'Data': datainfo,
                'Fold_Selector': fs,
                'Models': arcs,
                'Train_Results': tr_results_all,
                'Validation_Results': val_results_all,
        }

        # Save models
        where_to_save_model = log_dir + '/{}_{}_{}_start_time_{}.pickle'.format(Dataset, net_1, net_2, start_time)
        save_models(where_to_save_model, ModelSet, save=True)

    # ---------------------------- Testing the architecture ----------------------------
    if Stage is 'Testing':
        model_name = 'name_of_model'
        models_to_load = '../trained_models/{}/{}.pickle'.format(Dataset, model_name)
        models = load_model(models_to_load)
        index_best_model, element = max(enumerate(models['Validation_Results']), key=itemgetter(1))

        test_result, ts_snn_map = models['Models'][index_best_model].test(models['Models'][index_best_model].new_patterns,
                                                                        models['Data'], rs.ts_set)

        # Print final test performance results
        print('\n Best model index based on performance of the validation set: {}'.format(index_best_model))
        print('\n Best model accuracies \t Train: {}\t Validation: {}\t Test: {}'.format(models['Train_Results'][index_best_model],
                                                                                         models['Validation_Results'][index_best_model],
                                                                                         test_result))
