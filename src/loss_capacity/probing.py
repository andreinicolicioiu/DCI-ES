import torch.nn as nn
import torch.nn.functional as F
import torch
import rff
import numpy as np
import pdb

# one shared MLP predicts all the factors
# each factor is continuous
class Probe(nn.Module):
    def __init__(self, model, num_factors, num_hidden_layers=1, multiplier=16):
        super().__init__()
        self.model = model
        self.input_dim = model.latent_dim
        self.num_factors = num_factors
        self.multiplier = multiplier
        self.hidden_dim = self.input_dim
        self.num_hidden_layers = num_hidden_layers
        # multiplier <=0 -> linear probe
        if self.multiplier <= 0:
            self.num_hidden_layers = 0
        mlp_probe = []
        # for evaluating the performace at every layer of the probe
        self.probe_at_index = [nn.Identity()]
        
        if self.num_hidden_layers > 0:
            self.hidden_dim = self.multiplier * self.num_factors

            mlp_probe.append(nn.Linear(self.input_dim, self.hidden_dim))
            self.probe_at_index.append(nn.Sequential(*mlp_probe)) 
            mlp_probe.append(nn.ReLU(True))
            for i in range(self.num_hidden_layers - 1):
                mlp_probe.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                self.probe_at_index.append(nn.Sequential(*mlp_probe))
                mlp_probe.append(nn.ReLU(True))
                
        linear_probe = nn.Linear(self.hidden_dim, num_factors)
        mlp_probe.append(linear_probe)

        self.probe = nn.Sequential(*mlp_probe)
        # fix the parameters of the model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
    def count_parameters(model):
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_params

    def encode(self, x):
        features = self.model.encode(x)
        return features

    def forward(self, x):
        features = self.encode(x)
        pred_factors = self.probe(features)
        return pred_factors

    def forward_at_index(self, x, index=-1):
        features = self.encode(x)
        pred_factors = self.probe_at_index[index](features)
        return pred_factors
        
    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if mode == True:
            self.model.training = False # the model is always freezed and in eval mode
            # only the probe is in train mode
            for module in self.probe.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self
    def eval(self):
        return self.train(False)


# Each factor is predicted using its own MLP
class ProbeIndividual(nn.Module):
    def __init__(self, model, num_factors, num_hidden_layers=1, hidden_dim=0, factor_sizes=[2], 
            factor_discrete=[False], use_norm=False, use_dropout=False):
        super().__init__()
        self.model = model
        self.input_dim = model.latent_dim
        self.num_factors = num_factors
        self.factor_sizes = factor_sizes
        self.factor_discrete = factor_discrete
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        # self.capacity = 
        # self.multiplier = multiplier
        if self.hidden_dim == 0:
            self.num_hidden_layers = 0
            self.hidden_dim = self.input_dim

        if self.num_hidden_layers == 0:
            self.hidden_dim = self.input_dim

        self.probe = IndependentMLPs(num_factors, 
            self.input_dim, num_hidden_layers, 
            hidden_dim, factor_sizes, 
            factor_discrete,
            use_norm,
            use_norm
        )
        # fix the parameters of the model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        print(self.probe)
        print(f' Initialized probe with {self.num_hidden_layers} hidden layers of size {self.hidden_dim}')
        print(f' Total number of parameters {self.count_parameters()} parameters')
    
    def count_parameters(model):
        num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_params
    def encode(self, x):
        features = self.model.encode(x)
        return features
    def get_probe(self, enc):
        pred_factors = self.probe(enc)
        return pred_factors
    def forward_factors(self,x):
        # features = self.model.encode(x)
        features = self.encode(x)
        # features = F.relu(features)
        pred_factors = self.probe(features)
        all_pred_factors = []
        idx_factor_start = 0
        for i in range(len(self.factor_sizes)):
            if self.factor_discrete[i]:
                probs = pred_factors[:, idx_factor_start:idx_factor_start+self.factor_sizes[i]]
                out = F.softmax(probs, dim=-1) # this should be the final probs, so we apply softmax
                all_pred_factors.append(out)
                idx_factor_start += self.factor_sizes[i]
            else:
                out = pred_factors[:,idx_factor_start]
                all_pred_factors.append(out)
                idx_factor_start += 1
        return all_pred_factors

    def forward(self, x):
        # features = self.model.encode(x)
        features = self.encode(x)
        # features = F.relu(features)
        pred_factors = self.probe(features)
        return pred_factors

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if mode == True:
            self.model.training = False # the model is always freezed and in eval mode
            # only the probe is in train mode
            for module in self.probe.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self
    def eval(self):
        return self.train(False)


class IndependentMLPs(nn.Module):
    def __init__(self, num_factors, input_dim, num_hidden_layers=1, hidden_dim=0, factor_sizes=[2], 
            factor_discrete=[False], use_norm=False, use_dropout=False):
        super().__init__()
        self.input_dim = input_dim
        self.num_factors = num_factors
        self.factor_sizes = factor_sizes
        self.factor_discrete = factor_discrete
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        self.use_norm = use_norm
        self.use_dropout = use_dropout
        self.use_regularisation = self.use_norm
        if self.hidden_dim == 0:
            self.num_hidden_layers = 0
            self.hidden_dim = self.input_dim

        if self.num_hidden_layers == 0:
            self.hidden_dim = self.input_dim

        num_outputs = 0
        for i in range(num_factors):
            if self.factor_discrete[i]:
                # for each discrete factor we add K neurons
                num_outputs += self.factor_sizes[i]
            else:
                # for each continuous factor we add 1 neuron
                num_outputs += 1

        list_ind_mlps = []
        list_norms = []
        list_affine_scale = []
        list_affine_bias = []
        # input has size: B x D * num_factors x 1
        # each independent linear layer will act only on one group of features
        # this is equivalent with having num_factos separate networks
        if self.num_hidden_layers > 0:
            # list_ind_mlps.append(nn.Linear(self.input_dim, self.hidden_dim))
            group_independent_conv = nn.Conv1d(
                    in_channels = num_factors * self.input_dim,
                    out_channels = num_factors * self.hidden_dim, 
                    kernel_size = 1, 
                    groups = num_factors
                    )
            list_ind_mlps.append(group_independent_conv)

            # each group has an independent LayerNorm
            if self.use_norm:
                list_norms.append(nn.LayerNorm([self.hidden_dim,1], elementwise_affine=False))
                scale = torch.nn.Parameter(torch.Tensor(size=[self.hidden_dim * num_factors, 1]))
                bias = torch.nn.Parameter(torch.Tensor(size=[self.hidden_dim * num_factors, 1]))
                nn.init.constant_(bias, 0)
                nn.init.constant_(scale, 1)
                list_affine_scale.append(scale)
                list_affine_bias.append(bias)

            if not self.use_regularisation:
                list_ind_mlps.append(nn.ReLU(True))

            for i in range(self.num_hidden_layers - 1):
                # list_ind_mlps.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                group_independent_conv = nn.Conv1d(
                    in_channels = num_factors * self.hidden_dim,
                    out_channels = num_factors * self.hidden_dim, 
                    kernel_size = 1, 
                    groups = num_factors
                    )
                list_ind_mlps.append(group_independent_conv)
                if not self.use_regularisation:
                    list_ind_mlps.append(nn.ReLU(True))
                if self.use_norm:
                    list_norms.append(nn.LayerNorm([self.hidden_dim,1], elementwise_affine=False))
                    scale = torch.nn.Parameter(torch.Tensor(size=[self.hidden_dim * num_factors, 1]))
                    bias = torch.nn.Parameter(torch.Tensor(size=[self.hidden_dim * num_factors, 1]))
                    nn.init.constant_(bias, 0)
                    nn.init.constant_(scale, 1)
                    list_affine_scale.append(scale)
                    list_affine_bias.append(bias)

        if not self.use_regularisation:
            self.ind_mlps = nn.Sequential(*list_ind_mlps)
        else:
            self.ind_mlps = nn.ModuleList(list_ind_mlps)
            if self.use_norm:
                self.list_norms = nn.ModuleList(list_norms)
                self.list_affine_bias = nn.ParameterList(list_affine_bias)
                self.list_affine_scale = nn.ParameterList(list_affine_scale)

        ind_pred = []
        for i in range(num_factors):
            if self.factor_discrete[i]:
                predictor = nn.Linear(self.hidden_dim, self.factor_sizes[i])
            else:
                predictor = nn.Linear(self.hidden_dim, 1)
            ind_pred.append(predictor)

        self.ind_pred = nn.ModuleList(ind_pred)

    def forward(self, x):
        # x: B x D
        # repeat the input 
        # duplicate the input so that each conv group / (independent mlp) would have the same input
        x = x.tile(1,self.num_factors).unsqueeze(-1)
        # apply the independent MLPs on each group
        if not self.use_regularisation:
            x = self.ind_mlps(x)
        else:
            for i, ind_mlp in enumerate(self.ind_mlps):
                x = ind_mlp(x)
                if self.use_norm:
                    # LayerNorm should be applied for each independent mlp
                    x = x.view(x.shape[0], self.num_factors, -1,1)
                    x = self.list_norms[i](x)
                    x = x.view(x.shape[0], -1,1)
                    x = x * self.list_affine_scale[i] + self.list_affine_bias[i]
                if self.use_dropout:
                    x = F.dropout(x)
                x = F.relu(x)
        all_preds = []
        # output is the concatenation of the prediction of each factor
        for i in range(self.num_factors):
            x = x.view(x.shape[0], self.num_factors, -1)
            inp_i = x[:,i,:]
            pred_i = self.ind_pred[i](inp_i)
            all_preds.append(pred_i)

        all_preds = torch.cat(all_preds, dim=-1)
        return all_preds


class RFFProbeIndividual(nn.Module):
    def __init__(self, model, num_factors, num_hidden_layers=1, hidden_dim=0, factor_sizes=[2], 
            factor_discrete=[False], extra_hid_params=True, rff_sigma_gain=1.0, rff_sigma_scale='const'):
        super().__init__()
        self.model = model
        self.input_dim = model.latent_dim
        self.num_factors = num_factors
        self.factor_sizes = factor_sizes
        self.factor_discrete = factor_discrete
        self.num_hidden_layers = num_hidden_layers
        self.hidden_dim = hidden_dim
        # self.multiplier = multiplier
        if self.hidden_dim == 0:
            self.num_hidden_layers = 0
            self.hidden_dim = self.input_dim

        # if self.num_hidden_layers == 0:
        #     self.hidden_dim = self.input_dim
        if rff_sigma_scale == 'xavier':
            sigma = rff_sigma_gain * np.sqrt(2.0) / np.sqrt(self.hidden_dim + self.input_dim)
        elif rff_sigma_scale == 'const':
            sigma = rff_sigma_gain

        self.rff_encoding = rff.layers.GaussianEncoding(
                sigma=sigma, 
                input_size=self.input_dim, 
                encoded_size=self.hidden_dim // 2)

        self.probe = IndependentMLPs(num_factors, 
            self.hidden_dim, num_hidden_layers, 
            self.hidden_dim, factor_sizes, 
            factor_discrete,
            False
        )
        # fix the parameters of the model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        print(self.probe)
        print(f' Initialized probe with {self.num_hidden_layers} hidden layers of size {self.hidden_dim}')
        print(f' Total number of parameters {self.count_parameters()} parameters')
    
    def count_parameters(model):
        num_params =  sum(p.numel() for p in model.parameters() if p.requires_grad)
        return num_params

    def encode(self, x):
        features = self.model.encode(x)
        return features
    def get_probe(self, enc):
        rff_feats = self.rff_encoding(enc)
        pred_factors = self.probe(rff_feats)
        return pred_factors
    def forward_factors(self,x):
        features = self.model.encode(x)
        rff_feats = self.rff_encoding(features)
        pred_factors = self.probe(rff_feats)
        all_pred_factors = []
        idx_factor_start = 0
        for i in range(len(self.factor_sizes)):
            if self.factor_discrete[i]:
                probs = pred_factors[:, idx_factor_start:idx_factor_start+self.factor_sizes[i]]
                out = F.softmax(probs, dim=-1) # this should be the final probs, so we apply softmax
                all_pred_factors.append(out)
                idx_factor_start += self.factor_sizes[i]
            else:
                out = pred_factors[:,idx_factor_start]
                all_pred_factors.append(out)
                idx_factor_start += 1
        return all_pred_factors

    def forward(self, x):
        features = self.model.encode(x)
        rff_feats = self.rff_encoding(features)
        # rff_feats = self.encode(x)
        pred_factors = self.probe(rff_feats)
        return pred_factors

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        if mode == True:
            self.model.training = False # the model is always freezed and in eval mode
            # only the probe is in train mode
            for module in self.probe.children():
                module.train(mode)
        else:
            for module in self.children():
                module.train(mode)
        return self
    def eval(self):
        return self.train(False)


