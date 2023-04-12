import sys
sys.path.insert(0, './../')
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from PyTorchVAE.models.beta_vae import BetaVAE, SmallBetaVAE
from PyTorchVAE.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from PyTorchVAE.dataset import  DisentDatasets
from pytorch_lightning.plugins import DDPPlugin
from liftoff import parse_opts
from pathlib import Path

# -------
# import sys
# sys.path.insert(0, './../InDomainGeneralizationBenchmark/src/')
# import lablet_generalization_benchmark.evaluate_model as evaluate_model
# import lablet_generalization_benchmark.load_dataset as load_dataset
# import lablet_generalization_benchmark.model as models
# from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
# from loss_capacity.train_model import train_test_model
# from loss_capacity.probing import Probe
# datasets and evals



import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
# models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.train_model import train_test_model
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset
from loss_capacity.probing import Probe

import timm

import torch
import pickle
import json
import pdb


def run(params):
    """ Entry point for liftoff. """
    
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))

    config = params.vae

    print(f'out dir: {params.out_dir}')
        
    tb_logger =  TensorBoardLogger(save_dir=params.out_dir,
                                name=config.model_params.name)

    # For reproducibility
    seed_everything(config.exp_params.manual_seed * params.run_id, True)

    # TODO: maybe do smt more generic
    # model = vae_models[config.model_params.name](**configmodel_params)

    if config.model_params.model_type == 'small':
        model = SmallBetaVAE(**config.model_params.__dict__)
    elif config.model_params.model_type == 'big':
        model = BetaVAE(**config.model_params.__dict__)

    experiment = VAEXperiment(model,
                            config.exp_params)

    # if restore:
    #     ckpt = params.out_dir + '/BetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False
    dataloader_train = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='train_without_val',
        dataset_path=params.dataset_path, 
        batch_size=params.probe.batch_size, 
        num_workers=params.num_workers,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        data_fraction=1.0
    )

    dataloader_val = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='val',
        dataset_path=params.dataset_path, 
        batch_size=params.probe.batch_size, 
        num_workers=params.num_workers,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=False
    )

    dataloader_test = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='test',
        dataset_path=params.dataset_path, 
        batch_size=params.probe.batch_size, 
        num_workers=params.num_workers,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=False
    )
    
    data = DisentDatasets(
        train_batch_size=config.data_params.train_batch_size,
        val_batch_size= config.data_params.val_batch_size,
        test_batch_size= config.data_params.val_batch_size,
        train_dataset=dataloader_train.dataset,
        val_dataset=dataloader_val.dataset,
        test_dataset=dataloader_test.dataset,
        num_workers=config.data_params.num_workers,
        pin_memory=len(config.trainer_params.gpus) != 0)

    data.setup()
    savename = params.representation_dataset_path + f'/{params.dataset}_{params.model_type}_{params.name}'
    Path(savename).mkdir(parents=True, exist_ok=True)  

    ckeckpoint_path = savename + '.pt'
    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(#save_top_k=1, 
                                        every_n_epochs=config.trainer_params.max_epochs // 10,
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    strategy=DDPPlugin(find_unused_parameters=False),
                    **config.trainer_params.__dict__)

    for stage in ['val', 'test']:
        Path(f"{tb_logger.log_dir}/Samples_{stage}").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/Reconstructions_{stage}").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/InputImages_{stage}").mkdir(exist_ok=True, parents=True)
    
    if not os.path.exists(ckeckpoint_path):

        print(f"======= Training {config.model_params.name} =======")
        runner.fit(experiment, datamodule=data)

        runner.validate(experiment, datamodule=data)
        runner.test(experiment, datamodule=data)

        model.eval()

        Path(params.out_dir).mkdir(parents=True, exist_ok=True)  
        torch.save(model.state_dict(), savename+'.pt')
    else:
        ckpt = torch.load(ckeckpoint_path)
        model.load_state_dict(ckpt)
        model.eval()
        runner.validate(experiment, datamodule=data)
        runner.test(experiment, datamodule=data)

    # save the representations of final model
    if params.save_representation_datasets:
        device = 'cuda'
        model = model.to(device)
        save_representation_dataset(device, model, dataloader_train.dataset, f'{savename}_dataset_train_without_val')
        save_representation_dataset(device, model, dataloader_test.dataset, f'{savename}_dataset_test')
        save_representation_dataset(device, model, dataloader_val.dataset, f'{savename}_dataset_val')
    
if __name__ == "__main__":
    run(parse_opts())

