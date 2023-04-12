import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset import VAEDataset, DisentDatasets
from pytorch_lightning.plugins import DDPPlugin

# -------
import sys
sys.path.insert(0, './../InDomainGeneralizationBenchmark/src/')
import lablet_generalization_benchmark.evaluate_model as evaluate_model
import lablet_generalization_benchmark.load_dataset as load_dataset
import lablet_generalization_benchmark.model as models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.train_model import train_test_model
from loss_capacity.probing import Probe
import timm

import torch
import pickle
import json
import pdb


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

                    
parser.add_argument("--name", type=str, default='model')
parser.add_argument("--model_dir", type=str, default='./models_dir/tmp/')
parser.add_argument("--dataset", type=str, default='dsprites')


parser.add_argument(
    "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
)
parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 14)")
parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
# parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
parser.add_argument(
    "--log-interval",
    type=int,
    default=10,
    metavar="N",
    help="how many batches to wait before logging training status",
)
parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
# model parameters
parser.add_argument("--model_type", type=str, default='noisy_labels',
        help='model used for obtaining the representations chose from [raw_data/ noisy_labels / random_linear_mix_labels / noisy_uniform_mix_labels / conv_net]')
# model type: raw_data/ noisy_labels / random_linear_mix_labels / noisy_uniform_mix_labels / conv_net
# parser.add_argument("--supervised_model", action="store_true", default=False, help="train the model in a supervised way")
parser.add_argument("--supervision", type=str, default='none',
        help='supervision of the model [none / supervised]')
parser.add_argument("--pretrained", type=str, default='no',
        help='supervision of the model [no / yes]')

parser.add_argument("--noise_std", type=float, default=0.1, help="noise in models that use noisy labels as input")



# probe parameters
parser.add_argument("--probe_hidden_layers", type=int, default=2, help="number of hidden layers in the probe MLP (default: 1)")
parser.add_argument("--probe_hidden_multiplier", type=int, default=16, help="size of the hidden layer (multiplier x num_factors) (default: 16)")





args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

print(config)

tb_logger =  TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                               name=config['model_params']['name'],)

# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
experiment = VAEXperiment(model,
                          config['exp_params'])

data = DisentDatasets(**config["data_params"], pin_memory=len(config['trainer_params']['gpus']) != 0)

data.setup()
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2, 
                                     dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                     monitor= "val_loss",
                                     save_last= True),
                 ],
                 strategy=DDPPlugin(find_unused_parameters=False),
                 **config['trainer_params'])


Path(f"{tb_logger.log_dir}/Samples").mkdir(exist_ok=True, parents=True)
Path(f"{tb_logger.log_dir}/Reconstructions").mkdir(exist_ok=True, parents=True)


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)



device = 'cuda'
number_targets = len(data.train_dataloader().dataset._factor_sizes)

# create probe
probe = Probe(model, 
    num_factors=number_targets,
    num_hidden_layers=args.probe_hidden_layers,
    multiplier=args.probe_hidden_multiplier
)
probe = probe.to(device)

print(f'Initialising probe with {probe.count_parameters()} parameters')
# train the probe
probe, probe_test_loss = train_test_model(args, model=probe, 
    dataloader_train=data.train_dataloader(), dataloader_test=data.val_dataloader())

probe.eval()
def probe_fn(images):
    representation = probe(torch.tensor(images).to(device))
    return representation.detach().cpu().numpy()

# TODO: idealy we would want to use dev split for model selection and test for the final evaluation
# right now we use the model at the end of the training...
scores_probe = evaluate_model.evaluate_model(probe_fn, data.train_dataloader())
print(f'Scores Train probe: {scores_probe}')

scores_probe = evaluate_model.evaluate_model(probe_fn, data.val_dataloader())
print(f'Scores Test probe: {scores_probe}')

results = {}
num_probe_params = probe.count_parameters()
id=f'model_{args.name}_type_{args.model_type}_probe_params_{num_probe_params}'

results['id'] = id
results['model_type'] = args.model_type
results['num_params'] = num_probe_params
results['mse'] = scores_probe['mse']
results['rsquared'] = scores_probe['rsquared']
results['params'] = args
with open(f'{args.model_dir}/results.pkl', 'wb') as fp:
    pickle.dump(results, fp)

json_results = { key : str(val) for key, val in results.items()}

with open(f'{args.model_dir}/results.json', 'w') as fp:
    json.dump(json_results, fp)
