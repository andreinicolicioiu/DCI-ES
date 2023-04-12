
import os
import numpy
from argparse import Namespace
import numpy as np
import torch

from termcolor import colored as clr
import pdb


def list2tuple_(opt: Namespace) -> Namespace:
    """Deep (recursive) transform from Namespace to dict"""
    for key, value in opt.__dict__.items():
        name = key.rstrip("_")
        if isinstance(value, Namespace):
            setattr(opt, name, list2tuple_(value))
        else:
            if isinstance(value, list):
                setattr(opt, name, tuple(value))
    return opt


def config_to_string(
    cfg: Namespace, indent: int = 0, color: bool = True, verbose: bool = False
) -> str:
    """Creates a multi-line string with the contents of @cfg."""

    text = ""
    for key, value in cfg.__dict__.items():
        if key.startswith("__") and not verbose:
            # censor some fields
            pass
        else:
            ckey = clr(key, "yellow", attrs=["bold"]) if color else key
            text += " " * indent + ckey + ": "
            if isinstance(value, Namespace):
                text += "\n" + config_to_string(value, indent + 2, color=color)
            else:
                cvalue = clr(str(value), "white") if color else str(value)
                text += cvalue + "\n"
    return text



def get_representations_split(model, dataset, device, num_samples=None, shuffle=True, batch_size =  128):
    if num_samples == None:
        num_samples = len(dataset)
    elif num_samples > len(dataset):
        num_samples = len(dataset)

    dataloader =  torch.utils.data.DataLoader(dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1)

    inputs, targets = next(iter(dataloader))
    # feats = model.encode(inputs.to(device))
    feats = model(inputs.to(device))

    len_datasplits = (num_samples // batch_size + 1)  * batch_size
    all_feats = np.zeros((len_datasplits, feats.shape[1])).astype(np.float32)
    all_targets = np.zeros((len_datasplits, targets.shape[1])).astype(np.float32)

    # print(f'dataloader len: {len(dataset)}')
    data = []
    for idx, (data, target) in enumerate(dataloader):
        if idx * batch_size > num_samples:
             break
        data = data.to(device)
        # output = model.encode(data)
        output = model(data)
        output = output.detach().cpu().numpy()
        target = target.detach().cpu().numpy()
        
        all_feats[idx * batch_size: idx * batch_size + output.shape[0]] = output
        all_targets[idx * batch_size: idx * batch_size + target.shape[0]] = target

    
    all_feats = all_feats[:num_samples]
    all_targets = all_targets[:num_samples]

    return all_feats, all_targets

# TODO: use get_representations_split
def save_representation_dataset(device, model, dataset, path):
        print(f'saving representation dataset at: {path}')
        batch_size = 256
        dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4)

        inputs, targets = next(iter(dataloader))
        feats = model.encode(inputs.to(device))
        all_feats = np.zeros((len(dataset), feats.shape[1])).astype(np.float32)
        all_targets = np.zeros((len(dataset), targets.shape[1])).astype(np.float32)
        print(f'dataloader len: {len(dataset)}')
        data = []
        for idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            output = model.encode(data)
            output = output.detach().cpu().numpy()
            target = target.detach().cpu().numpy()
            
            all_feats[idx * batch_size: (idx + 1 ) * batch_size] = output
            all_targets[idx * batch_size: (idx + 1 ) * batch_size] = target

        # TODO: replace prev code by call to get_representations_split
        # should we create the array at the begening??
        np.save(path + '_feats.npy', all_feats)
        np.save(path + '_targets.npy', all_targets)

class RSquaredPytorch:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        all_possible_targets = dataloader.dataset.normalized_targets
        self.variance_per_factor = (
                (all_possible_targets -
                 all_possible_targets.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
  

        self.num_factors = self.variance_per_factor.shape[0]
        self.reset()
    def reset(self):
        self.diff = 0
        self.num_points = 0
    def acum_stats(self, predictions, targets):
        diff = (predictions - targets) ** 2
        self.diff += diff.shape[0] * diff.mean(dim=0)
        self.num_points += diff.shape[0]
    def get_scores(self):
        mse_loss_per_factor = self.diff / self.num_points
        mse_loss_per_factor = mse_loss_per_factor.detach().cpu().numpy()
        rsquared = 1 - mse_loss_per_factor / self.variance_per_factor

        scores = dict()
        scores['rsquared'] = rsquared.mean()
        scores['mse'] = mse_loss_per_factor.mean()

        factor_names = self.dataloader.dataset._factor_names
        for factor_index, factor_name in enumerate(factor_names):
            scores['rsquared_{}'.format(
                factor_name)] = rsquared[factor_index]
            scores['mse_{}'.format(factor_name)] = mse_loss_per_factor[factor_index]
        return scores


class MetricsPytorch:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        all_possible_targets = dataloader.dataset.normalized_targets
        self.variance_per_factor = (
                (all_possible_targets -
                 all_possible_targets.mean(axis=0, keepdims=True)) ** 2).mean(axis=0)
  

        self.num_factors = self.variance_per_factor.shape[0]
        self.factor_sizes = dataloader.dataset._factor_sizes
        self.factor_discrete = dataloader.dataset._factor_discrete

        self.reset()
    def reset(self):
        self.scores = [0 for i in range(self.num_factors)] # TODO: maybe replace it with torch.zeros that needs to be on device...
        self.num_points = 0
    def acum_stats(self, predictions, targets):
        idx_factor_start = 0
        eps = 0.00001
        for i in range(len(self.factor_sizes)):
            if self.factor_discrete[i]:
                target_index = targets[:,i] * (self.factor_sizes[i] - 1) + eps # target is already normalised
                target_index = target_index.type(torch.int64)

                one_hot = predictions[:, idx_factor_start:idx_factor_start+self.factor_sizes[i]]
                correct = torch.sum(one_hot.argmax(dim=1) == target_index)
                self.scores[i] += correct
                idx_factor_start += self.factor_sizes[i]
            else:
                l2_diff = (predictions[:,idx_factor_start] -  targets[:,i]) ** 2
                self.scores[i] += l2_diff.shape[0] * l2_diff.mean(dim=0)
                idx_factor_start += 1
            
        self.num_points += predictions.shape[0]
    def get_scores(self):
        
        scores_per_factor_ = torch.hstack(self.scores) / self.num_points
        scores_per_factor = scores_per_factor_.detach().cpu().numpy()
        for i in range(len(self.factor_sizes)):
            if not self.factor_discrete[i]:
                scores_per_factor[i] = 1 - scores_per_factor[i] / self.variance_per_factor[i]
        scores_d = dict()
        scores_d['rsquared'] = scores_per_factor.mean()
        scores_d['mse'] = 0 # we could just add the continuous error but it does't seem nenessary

        factor_names = self.dataloader.dataset._factor_names
        for factor_index, factor_name in enumerate(factor_names):
            scores_d['rsquared_{}'.format(
                factor_name)] = scores_per_factor[factor_index]
            scores_d['mse_{}'.format(factor_name)] = 0# mse_loss_per_factor[factor_index]
        return scores_d
