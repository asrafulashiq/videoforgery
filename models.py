from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11
from torchvision import transforms
from torchvision import models
import numpy as np


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet11(pretrained=True)

    def forward(self, x):
        x = self.unet(x)
        return x


def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.)


class Model_comp(nn.Module):
    """model to compare two frames with image patch
    """
    def __init__(self, size=(224, 224), drop_rate=0.3):
        super().__init__()
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        # create resnet block
        resnet = models.resnet34(pretrained=True)
        self.res_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # feature extractor with resnet

        # linear layer
        self.linear = nn.Sequential(
            nn.Linear(3 * 512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1),
        )

        self.linear.apply(init_weights)

    def forward(self, x):
        # x shape: (B, 3, 3, 224, 224)
        B, _, _, h, w = x.shape
        x = x.view(B * 3, 3, h, w)

        x_feat = self.res_extractor(x)
        x_feat = x_feat.view(B, 3 * 512)

        y = self.linear(x_feat)

        return y

