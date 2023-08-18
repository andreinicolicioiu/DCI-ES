
import glob
import pickle
import os.path
import numpy as np
import pdb
import sys
sys.path.insert(0, './../')
from metrics.explitcitness import compute_explitcitness, Explitcitness

import argparse
parser = argparse.ArgumentParser(description='Domain generalization')
parser.add_argument('--results_dir', type=str)

args = parser.parse_args()

dataset = 'dsprites' #  'mpi3d' /  'cars3d' / 'dsprites'
plot_probe =  'rf' # 'mlp' / 'rf' / 'rff'
num_seeds = 2

all_D, all_C, all_I, all_E = [], [], [], []
for seed_no in range(num_seeds):
    # we asume that the folder structure is folder_one_probe/seed_no/ 
    path_folders=f'{args.results_dir}/*/{seed_no}/'
    folders = glob.glob(path_folders)

    num_seeds = 1

    # TODO: create per_dataset num_factors, per_factor values etc..
    baseline_loss = {
            'mpi3d': [5.0 / 6, 5.0 / 6, 1.0, 1.0, 2.0 / 3, 1.0, 1.0],
            'cars3d': [1.0, 1.0, 182.0 / 183],
            'dsprites': [2.0 / 3, 1.0, 1.0, 1.0, 1.0]
        }
    results_capacity = {}
    for folder in folders:
        piclkle_name = folder + '/results.pkl'
        if os.path.exists(piclkle_name):
            with open(piclkle_name, 'rb') as f:
                results = pickle.load(f)
            id = results['id']
            test_rsquared_acc = results['test_rsquared_acc']#.cpu().numpy()
            val_rsquared_acc = results['val_rsquared_acc']#.cpu().numpy()

            probe_capacity = results['capacity']

            if results['dci_mlp'] is not None:
                dci_scores = results['dci_mlp']
            elif 'dci_trees' in results:
                dci_scores = results['dci_trees']

            results_capacity[probe_capacity] = [(probe_capacity, 1.0 - val_rsquared_acc, 1.0 - test_rsquared_acc, dci_scores)]
            results_capacity[probe_capacity] = [(probe_capacity, 1.0 - val_rsquared_acc, 1.0 - test_rsquared_acc, dci_scores)]
            # print(f'{piclkle_name}: cap: {probe_capacity} Info: {1.0 - test_rsquared_acc} DCI {dci_scores}')

    capacities = []
    d_scores = []
    c_scores = []
    i_scores = []
    r_scores = []

    for capacity in sorted(results_capacity.keys()):
        capacities.append(capacity)
        r_scores.append(results_capacity[capacity][0][2])
        if 'disentanglement' in results_capacity[capacity][0][3]:
            d_scores.append(results_capacity[capacity][0][3]['disentanglement'])
            c_scores.append(results_capacity[capacity][0][3]['completeness'])
        else:
            d_scores.append(results_capacity[capacity][0][3]['sage_disentanglement'])
            c_scores.append(results_capacity[capacity][0][3]['sage_completeness'])
        i_scores.append(results_capacity[capacity][0][3]['informativeness_test'])

    capacities = np.array(capacities)
    d_scores = np.array(d_scores)
    c_scores = np.array(c_scores)
    i_scores = np.array(i_scores)
    r_scores = np.array(r_scores).T # num_factors x num_probes

    num_factors = r_scores.shape[0]
    per_fact_exp = [Explitcitness() for _ in range(num_factors)]

    for f_i in range(num_factors):
        xx = capacities
        if not plot_probe == 'rf':
            xx = np.log(xx)
        yy = r_scores[f_i]
        name = f'model_factor_{f_i}'
        per_fact_exp[f_i].add_curve(xx, yy,
            baseline_loss[dataset][f_i],
            name=name)

    per_fact_E = np.zeros(num_factors)

    for f_i in range(num_factors):
        all_Exp = per_fact_exp[f_i].get_explitcitness(debug=False)
        for name, E in all_Exp.items():
            factor_i = int(name.split('_')[-1])
            per_fact_E[factor_i] = E


    explicitness = per_fact_E.mean()
    # print(f'Explicitness: {explicitness} ')
    best_info_ind = np.argmax(i_scores)
    best_info = i_scores[best_info_ind]
    best_d = d_scores[best_info_ind]
    best_c = c_scores[best_info_ind]

    all_D.append(best_d)
    all_C.append(best_c)
    all_I.append(best_info)
    all_E.append(explicitness)

all_D = np.array(all_D)
all_C = np.array(all_C)
all_I = np.array(all_I)
all_E = np.array(all_E)
print(f'D: {all_D.mean():.3f} ± {all_D.std():.3f} \
      C: {all_C.mean():.3f} ± {all_C.std():.3f} \
      I {all_I.mean():.3f} ± {all_I.std():.3f} \
      E: {all_E.mean():.3f} ± {all_E.std():.3f} ')


