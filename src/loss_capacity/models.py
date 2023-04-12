
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class AbstractModel(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1):
        super().__init__()
        self.latent_dim = -1

    def encode(self, x):
        flat = torch.flatten(x, 1)
        return flat

    def forward(self, x):
        return self.encode(x)

class ConvNet(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(number_of_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, self.latent_dim, 4, 1),  # B, 256,  1,  1
        )
        self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, x):
        x = self.encode(x)
        pred = self.predictor(x)
        return pred

import timm
class ResNet18(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, latent_dim=256, pretrained=True):
        super().__init__()
        
        self.encoder = timm.create_model('resnet18', 
            pretrained=pretrained, 
            in_chans=number_of_channels,
            num_classes=0)# the last layer of resnet would be identity
        self.latent_dim = self.encoder.feature_info[-1]['num_chs']
        self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, x):
        x = self.encode(x)
        pred = self.predictor(x)
        return pred

class RawData(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        # self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        flat = torch.flatten(x, 1)
        return flat

    def forward(self, x):
        return self.encode(x)

