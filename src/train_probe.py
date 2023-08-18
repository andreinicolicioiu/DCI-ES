import sys
# sys.path.insert(0, './../src')
sys.path.insert(0, './../')
from liftoff import parse_opts
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.model as models
from loss_capacity.models import ConvNet, RawData, ResNet18
from loss_capacity.train_model import train_test_probe, evaluate_probe
from loss_capacity.train_model_random_forest import train_test_random_forest
from loss_capacity.probing import Probe, ProbeIndividual, RFFProbeIndividual
from loss_capacity.utils import list2tuple_, config_to_string
# from metrics.get_metric import get_probe_dci, get_dci
# from metrics.compute_metrics import compute_disentanglement_metric
import timm

import torch
import pickle
import json
import os
import pdb

def run(params):
    """ Entry point for liftoff. """
    DEBUG_FLAG = False

    params = list2tuple_(params)
    print(config_to_string(params))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cuda or cpu

    sage_group_pixels = True if 'raw' in params.model_type else False
    print(f'WARNING: hardcodding sage_group pixels={sage_group_pixels}')


    if 'none' in  params.probe.max_leaf_nodes or 'None'  in  params.probe.max_leaf_nodes:
        params.probe.max_leaf_nodes = None

    torch.multiprocessing.set_sharing_strategy('file_system')

    cached_data_path = os.path.join(params.representation_dataset_path, 
        params.representation_dataset_name)

    number_of_channels = 1 if params.dataset == 'dsprites' else 3

    if params.cached_representations and ('resnet' in params.model_type or 'vae' in params.model_type ):
        # create dataloader from the saved representations
        dataloader_train = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,
            mode='train_without_val',
            dataset_path=cached_data_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            data_fraction=params.probe.data_fraction if DEBUG_FLAG == False else 256
        )

        dataloader_val = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,
            mode='val',
            dataset_path=cached_data_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False,
            data_fraction=1.0 if DEBUG_FLAG == False else 256
        )

        dataloader_test = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,
            mode='test' if DEBUG_FLAG == False else 'val',
            dataset_path=cached_data_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False,
            data_fraction=1.0 if DEBUG_FLAG == False else 256
        )
        inputs, targets = next(iter(dataloader_test))
        # since we have precomputed the representations, here we use as model the identity function
        model = RawData(latent_dim=inputs.shape[-1])
        model = model.to(device)
        number_targets = len(dataloader_train.dataset._factor_sizes)

    print(f'Loaded train dataset with size: {len(dataloader_train.dataset)}')
    print(f'Loaded val dataset with size: {len(dataloader_val.dataset)}')
    print(f'Loaded test dataset with size: {len(dataloader_test.dataset)}')
    

    model.eval()

    # crate tensorboard writter
    writer = SummaryWriter(log_dir=params.out_dir)
    # create probe
    probe = None
    if 'MLP' in params.probe.type:
    # if params.probe.type == 'MLP_reg_cl_ind':
        probe = ProbeIndividual(model, 
            num_factors=number_targets,
            num_hidden_layers=params.probe.hidden_layers,
            hidden_dim=params.probe.hidden_dim,
            factor_sizes=dataloader_train.dataset._factor_sizes,
            factor_discrete=dataloader_train.dataset._factor_discrete,
            use_norm=params.probe.use_norm,
            use_dropout=params.probe.use_dropout
        )
        probe = probe.to(device)
    # elif params.probe.type == 'RFF_MLP_reg_cl_ind':
    elif 'RFF' in params.probe.type:
        probe = RFFProbeIndividual(model, 
            num_factors=number_targets,
            num_hidden_layers=params.probe.hidden_layers,
            hidden_dim=params.probe.hidden_dim,
            factor_sizes=dataloader_train.dataset._factor_sizes,
            factor_discrete=dataloader_train.dataset._factor_discrete,
            rff_sigma_gain=params.probe.rff_sigma_gain,
            rff_sigma_scale=params.probe.rff_sigma_scale
        )
        probe = probe.to(device)

    # train the probe
    if probe is not None:
        probe.train()
        probe.model.eval() # the model is freezed, so it should stay in eval mode


    resume = True
    dci_scores_trees = None
    dci_scores_val = None
    dci_scores = None
    probe_test_loss=0
    probe_val_score = probe_test_score =  -1
    ckeckpoint_path = f'{params.out_dir}/best_probe_model.pt'
    ckeckpoint_path = f'{params.out_dir}/best_probe_model.pt'
    if not resume or os.path.exists(ckeckpoint_path) == False:
        if 'MLP' in params.probe.type or 'RFF' in params.probe.type:
            probe, probe_test_loss, probe_test_score, probe_val_score = train_test_probe(model=probe, 
                dataloader_train=dataloader_train, 
                dataloader_val=dataloader_val, 
                dataloader_test=dataloader_test, 
                seed = params.seed * params.run_id, 
                lr = params.probe.lr, 
                weight_decay = params.probe.weight_decay,
                optim_steps=params.probe.optim_steps,
                epochs = params.probe.epochs, 
                log_interval =  params.log_interval, save_model = params.save_model,
                train_name='probe_mlp',
                tb_writer=writer, eval_model=True, savefolder=params.out_dir)
            num_probe_params = probe.count_parameters()
        elif 'RandomForest' in params.probe.type:
            capacity, probe_train_score, probe_val_score, probe_test_score, dci_scores_trees = train_test_random_forest(
                    model=model, 
                    dataloader_train=dataloader_train, 
                    dataloader_val=dataloader_val, 
                    dataloader_test=dataloader_test, 
                    method='rf',
                    max_leaf_nodes=params.probe.max_leaf_nodes,
                    max_depth=params.probe.max_depth,
                    num_trees=params.probe.num_trees,
                    seed=params.seed * params.run_id, 
                    lr = params.probe.rf_lr,
                    epochs = params.probe.epochs, 
                    log_interval =  params.log_interval, save_model = params.save_model,
                    train_name='probe_trees',
                    tb_writer=writer, eval_model=True, savefolder=params.out_dir,
                    device=device,
                    data_fraction=params.probe.data_fraction,
                    use_sage=False
                    )
            num_probe_params = 0 # dummy, there are no learnable parameters in random-forrest
            probe_test_loss = 0
    else:
        if os.path.exists(ckeckpoint_path):
            ckpt = torch.load(ckeckpoint_path)
            probe.load_state_dict(ckpt)
        num_probe_params = probe.count_parameters()

    if probe is not None:
        probe.eval()


    from pathlib import Path
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    id=f'model_{params.name}_type_{params.model_type}_probe_params_{num_probe_params}'
    
    results['id'] = id
    results['model_type'] = params.model_type
    results['num_params'] = num_probe_params // number_targets

    if 'MLP' in params.probe.type:
        results['capacity'] = num_probe_params // number_targets
    elif 'RFF' in params.probe.type:
        results['capacity'] = params.probe.hidden_dim 
    else:
        results['capacity'] = capacity

    results['mse'] = probe_test_loss 
    results['val_rsquared_acc'] = probe_val_score
    results['test_rsquared_acc'] = probe_test_score
    results['dci_trees'] = dci_scores_trees
    results['params'] = params
    # sage computations might last so we save everything else before
    if 'MLP' in params.probe.type or 'RFF' in params.probe.type:
        probe, probe_val_loss, probe_val_score, dci_scores_val = evaluate_probe(
                model=probe,
                dataloader_test=dataloader_val, 
                seed = params.seed * params.run_id, #+ seed_bias,
                log_interval = params.log_interval,
                train_name='eval_probe_val',
                tb_writer=writer,
                compute_dci=True,
                group_pixels=sage_group_pixels)
                
        probe, probe_test_loss, probe_test_score, dci_scores = evaluate_probe(
                model=probe,
                dataloader_test=dataloader_test, 
                seed = params.seed * params.run_id, #+ seed_bias,
                log_interval =  params.log_interval,
                train_name='eval_probe_test',
                tb_writer=writer,
                compute_dci=True,
                group_pixels=sage_group_pixels)
    
    results['dci_mlp_val'] = dci_scores_val
    results['dci_mlp'] = dci_scores
    results['val_rsquared_acc'] = probe_val_score
    results['test_rsquared_acc'] = probe_test_score
    print(f'saving results in {params.out_dir}/results.pkl')

    with open(f'{params.out_dir}/results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    json_results = { key : str(val) for key, val in results.items()}

    with open(f'{params.out_dir}/results.json', 'w') as fp:
        json.dump(json_results, fp)

if __name__ == "__main__":
    run(parse_opts())
