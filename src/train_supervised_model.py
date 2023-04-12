import sys
sys.path.insert(0, './../')

import torch
import numpy as np
import timm
# datasets and evals
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
# models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.train_model import train_test_model
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset
from loss_capacity.probing import Probe
# args and save results
from torch.utils.tensorboard import SummaryWriter
from liftoff import parse_opts
from pathlib import Path
import pickle
import json
import pdb

def run(params):

    """ Entry point for liftoff. """
    
    params = list2tuple_(params)
    print(config_to_string(params))

    print(f'out dir: {params.out_dir}')
    # print(f'starting model: {params.model_type}')
    # return 0
    # load appropiate dataset depending on model type
    # models based on noisy (mix of) labels loads only the labels dataset
    number_of_channels = 1 if params.dataset == 'dsprites' else 3
    print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False

    dataloader_train = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='train_without_val',
        dataset_path=params.dataset_path, 
        batch_size=params.model.batch_size, 
        num_workers=10,
        standardise=True,
        imagenet_normalise=imagenet_normalise
    )

    dataloader_val = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='val',
        dataset_path=params.dataset_path, 
        batch_size=params.model.batch_size, 
        num_workers=10,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=False
    )

    dataloader_test = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='test',
        dataset_path=params.dataset_path, 
        batch_size=params.model.batch_size, 
        num_workers=10,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=False
    )

    number_targets = len(dataloader_train.dataset._factor_sizes)
    device = 'cuda'  # cuda or cpu
    # load model   
    if params.model_type == 'conv_net':
        model = ConvNet(
            number_of_classes=number_targets, 
            number_of_channels=number_of_channels
        )
    elif params.model_type == 'resnet18':
        # model = timm.create_model('resnet18', 
        #     pretrained=True, 
        #     in_chans=1,
        #     num_classes=number_targets)
        model = ResNet18(
            number_of_classes=number_targets, 
            number_of_channels=number_of_channels,
            pretrained=params.pretrained
        )

    model = model.to(device)

    # TODO
    # add pretrained models!!!
    # TODO: change flag name to training instead of supervision
    if params.supervision:
        writer = SummaryWriter(log_dir=params.out_dir)

        model, test_loss = train_test_model(model=model, 
                    dataloader_train=dataloader_train, 
                    dataloader_val=dataloader_val,
                    seed = params.seed,
                    lr = params.model.lr, optim_steps=params.model.optim_steps,
                    epochs = params.model.epochs, 
                    log_interval =  params.log_interval, save_model = params.save_model,
                    train_name='model',
                    tb_writer=writer, eval_model=True)

        # probe, probe_test_loss, probe_test_score, probe_val_score = train_test_probe(model=probe, 
        #     dataloader_train=dataloader_train, 
        #     dataloader_val=dataloader_val, 
        #     dataloader_test=dataloader_test, 
        #     seed = params.seed * params.run_id + seed_bias,
        #     lr = params.probe.lr, 
        #     weight_decay = params.probe.weight_decay,
        #     optim_steps=params.model.optim_steps,
        #     epochs = params.probe.epochs, 
        #     log_interval =  params.log_interval, save_model = params.save_model,
        #     train_name='probe_mlp',
        #     tb_writer=writer, eval_model=True, savefolder=params.out_dir,
        #     cross_ent_mult=params.probe.cross_ent_mult)

    model.eval()
    def model_fn1(images):
        representation = model(torch.tensor(images).to(device))
        return representation.detach().cpu().numpy()

    scores = evaluate_model.evaluate_model(model_fn1, dataloader_test)
    print(f'Scores: {scores}')

    savename = params.resnet_representation_dataset_path + f'/tmp_{params.dataset}_{params.model_type}_{params.name}'

    from pathlib import Path
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)  
    Path(savename).mkdir(parents=True, exist_ok=True)  

    torch.save(model.state_dict(), savename+'.pt')

    results = {}
    results['model_type'] = params.model_type
    results['mse'] = scores['mse']
    results['rsquared'] = scores['rsquared']
    results['params'] = params

    with open(f'{savename}_results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    json_results = { key : str(val) for key, val in results.items()}

    with open(f'{savename}_results.json', 'w') as fp:
        json.dump(json_results, fp)

    if params.save_representation_datasets:
        save_representation_dataset(device, model, dataloader_train.dataset, f'{savename}_dataset_train_without_val')
        save_representation_dataset(device, model, dataloader_val.dataset, f'{savename}_dataset_val')
        save_representation_dataset(device, model, dataloader_test.dataset, f'{savename}_dataset_test')



    return model
# def save_representation_dataset(device, model, dataset, path):
#         print(f'saving representation dataset at: {path}')
#         batch_size = 256
#         dataloader = torch.utils.data.DataLoader(dataset,
#             batch_size=batch_size,
#             shuffle=False,
#             num_workers=4)

#         inputs, targets = next(iter(dataloader))
#         feats = model.encode(inputs.to(device))
#         all_feats = np.zeros((len(dataset), feats.shape[1])).astype(np.float32)
#         all_targets = np.zeros((len(dataset), targets.shape[1])).astype(np.float32)
#         print(f'dataloader len: {len(dataset)}')
#         data = []
#         for idx, (data, target) in enumerate(dataloader):
#             data = data.to(device)
#             output = model.encode(data)
#             output = output.detach().cpu().numpy()
#             target = target.detach().cpu().numpy()
            
#             all_feats[idx * batch_size: (idx + 1 ) * batch_size] = output
#             all_targets[idx * batch_size: (idx + 1 ) * batch_size] = target

#         # should we create the array at the begening??
#         np.save(path + '_feats.npy', all_feats)
#         np.save(path + '_targets.npy', all_targets)

if __name__ == "__main__":
    run(parse_opts())
