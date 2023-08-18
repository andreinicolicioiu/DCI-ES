import numpy as np
import torch
import pdb
import os
# from sklearn.ensemble import GradientBoostingClassifier
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from metrics.dci import disentanglement, completeness
import sage

import xgboost as xgb
from loss_capacity.utils import get_representations_split
import time
def train_test_random_forest(
        model, dataloader_train, dataloader_val, dataloader_test,
        method='tree_ens_hgb',
        max_leaf_nodes = 200,
        max_depth=20,
        num_trees=100,
        seed = 0, lr = 0.001, epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        device=None,
        data_fraction=1.0,
        use_sage=False):

    sage_thresh = 0.2 #5
    sage_num_samples = 16 #4096
    # method 
    # tree_ens_rf
    # tree_ens_hgb
    # tree_ens_xgb

    print(f'learn {method} probes')
    print(f'Use trees with max depth: {max_depth}. max_leaf_nodes: {max_leaf_nodes}. num_trees: {num_trees}.')

    num_train   = len(dataloader_train.dataset)
    print(f'Train tree ensemble on: {num_train} samples')

    num_val     =  len(dataloader_val.dataset)
    num_test    =  len(dataloader_test.dataset)

    train_h, train_targets = get_representations_split(model, dataloader_train.dataset, device, len(dataloader_train.dataset))
    val_h, val_targets = get_representations_split(model, dataloader_val.dataset, device, num_val)
    test_h, test_targets = get_representations_split(model, dataloader_test.dataset, device, num_test)

    factor_sizes    = dataloader_train.dataset._factor_sizes
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_names    = dataloader_train.dataset._factor_names
    num_factors = len(factor_sizes)

    all_rsquared_train = np.zeros(num_factors)
    all_rsquared_test = np.zeros(num_factors)
    all_rsquared_val = np.zeros(num_factors)

    eps = 0.00001
    time_start = time.time()
    R = np.zeros((train_h.shape[1], num_factors))
    R_sage = np.zeros((train_h.shape[1], num_factors))

    # folder for saving probe models
    Path(savefolder+'/rf_models/').mkdir(parents=True, exist_ok=True)
    resume = True
    # use_sage = False
    for ind_fact in range(num_factors):
        savefile = f"{savefolder}/rf_models/rf_datasize_{num_train}_factor_{ind_fact}.joblib"
        savefile_sage = f"{savefolder}/rf_models/sage_rf_datasize_{num_train}_factor_{ind_fact}.joblib"

        if factor_discrete[ind_fact]:
            #classification
            train_t = (train_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
            val_t   = (val_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
            test_t  = (test_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
            if resume and os.path.exists(savefile):
                print(f'Load model from: {savefile}')
                model = joblib.load(savefile) 
            else:
                if 'xgb' in method:
                    model = xgb.XGBClassifier(verbosity=2,
                                max_depth=max_depth)
                elif 'hgb' in method:
                    model = HistGradientBoostingClassifier(verbose=1,
                                max_leaf_nodes=max_leaf_nodes,
                                max_depth=max_depth,
                                learning_rate=lr)
                elif 'rf' in method:
                    model = RandomForestClassifier(verbose=1,
                                max_leaf_nodes=max_leaf_nodes,
                                max_depth=max_depth,
                                n_estimators=num_trees,
                                random_state=seed,
                                n_jobs=-1)
                model.fit(train_h, train_t)
            # for classification this computes the mean accuracy
            all_rsquared_train[ind_fact]  = model.score(train_h, train_t)
            all_rsquared_val[ind_fact]  = model.score(val_h, val_t)
            all_rsquared_test[ind_fact]  = model.score(test_h, test_t)
            R[:,ind_fact] = np.abs(model.feature_importances_)  

            if use_sage:
                # Setup and calculate
                imputer = sage.MarginalImputer(model.predict_proba, test_h[:sage_num_samples])
                estimator = sage.PermutationEstimator(imputer, 'cross entropy')
                sage_values = estimator(test_h, test_t, batch_size=128, thresh=sage_thresh)
                sage_vals = np.abs(sage_values.values)  # TODO: these are not normalised. solved
                sage_vals = sage_vals / sage_vals.sum()
                R_sage[:,ind_fact] = sage_vals

        else:
            # regression
            train_t = train_targets[:, ind_fact]
            val_t   = val_targets[:, ind_fact]
            test_t  = test_targets[:, ind_fact]
            if resume and os.path.exists(savefile):
                print(f'Load model from: {savefile}')
                model = joblib.load(savefile) 
            else:
                if 'xgb' in method :
                    model = xgb.XGBRegressor(verbose=1,
                                max_depth=max_depth)
                elif 'hgb' in method:
                    model = HistGradientBoostingRegressor(verbose=1, 
                                max_leaf_nodes=max_leaf_nodes,
                                max_depth=max_depth,
                                learning_rate=lr)
                elif 'rf' in method:
                    model = RandomForestRegressor(verbose=1,
                                max_leaf_nodes=max_leaf_nodes,
                                max_depth=max_depth,
                                n_estimators=num_trees,
                                random_state=seed,
                                n_jobs=-1)
                
                model.fit(train_h, train_t)
            
            all_rsquared_train[ind_fact]  = model.score(train_h, train_t)
            all_rsquared_val[ind_fact]  = model.score(val_h, val_t)
            all_rsquared_test[ind_fact]  = model.score(test_h, test_t)
            R[:,ind_fact] = np.abs(model.feature_importances_)

            if use_sage:
                # Setup and calculate
                imputer = sage.MarginalImputer(model.predict, test_h[:sage_num_samples])
                estimator = sage.PermutationEstimator(imputer, 'mse')
                sage_values = estimator(test_h, test_t, batch_size=128, thresh=sage_thresh)
                sage_vals = np.abs(sage_values.values)  
                sage_vals = sage_vals / sage_vals.sum()
                R_sage[:,ind_fact] = sage_vals

        # save
        if not (resume and os.path.exists(savefile)):
            joblib.dump(model, savefile)
        
    total_time = time.time() - time_start

    print(f'\nTree ensemble train scores mean: {all_rsquared_train.mean()}')
    for i, name in enumerate(factor_names):
        print(f'{name}: {all_rsquared_train[i]}')

    print(f'\nTree ensemble validation scores mean: {all_rsquared_val.mean()}')
    for i, name in enumerate(factor_names):
        print(f'{name}: {all_rsquared_val[i]}')
    
    print(f'\nTree ensemble test scores mean: {all_rsquared_test.mean()}')
    for i, name in enumerate(factor_names):
        print(f'{name}: {all_rsquared_val[i]}')

    capacity = max_depth if max_depth is not None else -10 #max_leaf_nodes 

    for i in range(10):
        tb_writer.add_scalar(f'{train_name}-rsquared_acc/train', all_rsquared_train.mean(), capacity+i)
        tb_writer.add_scalar(f'{train_name}-rsquared_acc/val', all_rsquared_val.mean(), capacity+i)
        tb_writer.add_scalar(f'{train_name}-rsquared_acc/test', all_rsquared_test.mean(), capacity+i)
        tb_writer.add_scalar(f'{train_name}-time/time_training',total_time, capacity+i)
        for j in range(len(all_rsquared_val)):
            tb_writer.add_scalar(f'{train_name}-scores/val_{factor_names[j]}',all_rsquared_val[j], capacity+i)
            tb_writer.add_scalar(f'{train_name}-scores/test_{factor_names[j]}',all_rsquared_test[j], capacity+i)


    dci_scores = {}
    dci_scores["informativeness_train"] = all_rsquared_train.mean()
    dci_scores["informativeness_val"] = all_rsquared_val.mean()
    dci_scores["informativeness_test"] = all_rsquared_test.mean()
    dci_scores["disentanglement"] = disentanglement(R)
    dci_scores["completeness"] = completeness(R)

    if use_sage:
        dci_scores["sage_disentanglement"] = disentanglement(R_sage)
        dci_scores["sage_completeness"] = completeness(R_sage)


    for name, score in dci_scores.items():
        print(f'DCI: {name}: {score} ')
        for i in range(10):
            tb_writer.add_scalar(f'{train_name}-scores/DCI_trees_ens_{name}', score, max_depth)

    print(f'sage: sage_num_samples: {sage_num_samples}, sage_thresh: {sage_thresh}')

    print(f'Gini importance R')
    print(R)
    print(f'Sage importance R')
    print(R_sage)

    return capacity, all_rsquared_train, all_rsquared_val, all_rsquared_test, dci_scores

