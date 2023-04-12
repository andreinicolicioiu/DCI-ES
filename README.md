# DCI-ES

 This is the official repo of the paper [DCI-ES: An Extended Disentanglement Framework with Connections to Identifiability](https://openreview.net/forum?id=462z-gLgSht&noteId=dVL1YIoSbD)

The paper explores the idea of analysing the properties of a learned representation from the point of view of ease-of-use. The main tool to do this is the loss-capacity curves, as explained in the paper. For a given representation, we train a set of probes with increasing capacity that are learned (in a supervised way) to predict the latent factors of variations for a dataset. After training the set of probes, we can compute the proposed Explicitness score, as shown in the paper.

For computing the DCI-ES scores need 4 steps: 
 1) get pretrained representations or train representations on evaluation datasets 
 2) compute the representations of the evaluation dataset and save the results 
 3) train the sets of probes 
 4) compute DCI-ES scores.

### install dependencies
```pip install -r requirements_pip.txt```


## Datasets
We use [MPI3D](https://arxiv.org/abs/1906.03292) dataset for exaluation in this repo. 

## Train and save basic representations of the evaluation dataset
We can use the train_unsupervised_model.py to train a beta-VAE model on the evaluation datasets using the default parameters from [default_config_dev.yaml](https://github.com/andreinicolicioiu/loss_capacity/blob/iclr_ready/configs/default_config_dev.yaml).

```sh
liftoff train_unsupervised_model.py ../configs/default_config_dev.yaml
```

The previous script can be adapted to include any pretrained representation that we might want to test. As long as we save the vector representations into the same format as the previous script, we can use them for computing the DCI-ES scores. 

## Train a single probe
 Train a single probes with default parameters ([default_config_dev.yaml](https://github.com/andreinicolicioiu/loss_capacity/blob/iclr_ready/configs/default_config_dev.yaml)):

```sh
liftoff train_probe_clean.py ../configs/default_config_dev.yaml
```

We can change the hyperparameters of the probe, defined in the config file and train a new probe. 
The most important hyperparameter is the type of probe: MLPs, Random Fourier Features + Learned linear layer, Random Forest. This is selected by the flag: probe_type [MLP / RFF / RandomForest].

## Train set of probes
We use [liftoff](https://github.com/tudor-berariu/) to generate config files for different runs, and for managing a queue of experiments.

First we create a set of config files, each of for training one probe. We define in ```../configs/random_forest/default.yaml``` the default parameters used in all experiments of a set, and in ```../configs/random_forest/config.yaml``` the hyperparameters that we vary. 

```sh
liftoff-prepare ../configs/random_forest/ --runs-no 1 --results-path results/ --do
```
The previous command creates 10 runs (10 different seeds) for each hyperparameter configuration defined in ../configs/random_forest/config.yaml and save them in a results folder.

We run all experiments in the queue using the command:
```sh
liftoff train_probe.py  ./results/date_random_forest/  --gpus 0 --per-gpu 4 --procs-no 4
```
The previous command starts 4 runs in paralel on one GPU.


## Compute Explicitness score
After we have trained all probes of a certain type, we can compute the DCI-E scores using the following script:
```sh
python simple_gather.py --results_dir=results/date_random_forest
```

## In this repo we make use of the following projects:
 - SAGE: https://github.com/iancovert/sage
 - loaders for MPI3D from: https://github.com/bethgelab/InDomainGeneralizationBenchmark
 - Random Fourier Features implementation: https://github.com/jmclong/random-fourier-features-pytorch



## Citation
Please use the following BibTeX to cite our work.
```
@inproceedings{
eastwood2022dcies,
title={{DCI}-{ES}: An Extended Disentanglement Framework with Connections to Identifiability},
author={Eastwood, Cian and Nicolicioiu, Andrei Liviu and von K{\"u}gelgen, Julius and Keki{\'c}, Armin and Tr{\"a}uble, Frederik and Dittadi, Andrea and Sch{\"o}lkopf, Bernhard},
booktitle={International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=462z-gLgSht}
}
```
