import numpy as np
import warnings
from sage import utils
import time
import pdb

class Imputer:
    '''Imputer base class.'''
    def __init__(self, model):
        self.model = utils.model_conversion(model)

    def __call__(self, x, S):
        raise NotImplementedError


class DefaultImputer(Imputer):
    '''Replace features with default values.'''
    def __init__(self, model, values):
        super().__init__(model)
        if values.ndim == 1:
            values = values[np.newaxis]
        elif values[0] != 1:
            raise ValueError('values shape must be (dim,) or (1, dim)')
        self.values = values
        self.values_repeat = values
        self.num_groups = values.shape[1]

    def __call__(self, x, S):
        # Prepare x.
        if len(x) != len(self.values_repeat):
            self.values_repeat = self.values.repeat(len(x), 0)

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.values_repeat[~S]

        # Make predictions.
        return self.model(x_)


class MarginalImputer(Imputer):
    '''Marginalizing out removed features with their marginal distribution.'''
    def __init__(self, model, data):
        super().__init__(model)
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = data.shape[1]

        if len(data) > 1024:
            warnings.warn('using {} background samples may lead to slow '
                          'runtime, consider using <= 1024'.format(
                            len(data)), RuntimeWarning)

    def __call__(self, x, S):
        # pdb.set_trace()
        # Prepare x and S.
        n = len(x)
        x = x.repeat(self.samples, 0)
        S = S.repeat(self.samples, 0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = np.tile(self.data, (n, 1))

        # Replace specified indices.
        x_ = x.copy()
        x_[~S] = self.data_repeat[~S]

        # Make predictions.
        pred = self.model(x_)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)

import torch
class MarginalImputerPytorch(Imputer):
    '''Marginalizing out removed features with their marginal distribution.'''
    def __init__(self, model, data):
        super().__init__(model)
        self.data = data
        self.data_repeat = data
        self.samples = len(data)
        self.num_groups = data.shape[1]

        if len(data) > 1024:
            warnings.warn('using {} background samples may lead to slow '
                          'runtime, consider using <= 1024'.format(
                            len(data)), RuntimeWarning)

    def __call__(self, x, S):
        # pdb.set_trace()
        # Prepare x and S.
        n = len(x)
        x_py = torch.Tensor(x).to(self.data.device)
        x_py = x_py.repeat_interleave(self.samples, dim=0)

        S_py = torch.Tensor(S).to(self.data.device).bool()
        S_py = S_py.repeat_interleave(self.samples, dim=0)

        # Prepare samples.
        if len(self.data_repeat) != self.samples * n:
            self.data_repeat = torch.tile(self.data, (n, 1))

        x_py[~S_py] = self.data_repeat[~S_py]
        pred = self.model(x_py)
        pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        return np.mean(pred, axis=1)


        # x = x.repeat(self.samples, 0)
        # S = S.repeat(self.samples, 0)

        # # Prepare samples.
        # if len(self.data_repeat) != self.samples * n:
        #     self.data_repeat = np.tile(self.data, (n, 1))

        # # Replace specified indices.
        # x_ = x.copy()
        # x_[~S] = self.data_repeat[~S]

        # # Make predictions.
        # pred = self.model(x_)
        # pred = pred.reshape(-1, self.samples, *pred.shape[1:])
        # return np.mean(pred, axis=1)
