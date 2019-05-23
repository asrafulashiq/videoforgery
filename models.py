from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = UNet11(pretrained=True)

    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)
        return x