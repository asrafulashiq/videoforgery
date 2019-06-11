from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11, AlbuNet, UNet11_two_branch_out
from torchvision import transforms
from torchvision import models
import numpy as np
from modeling import deeplab



def get_model(type, pretrained=True, num_classes=1):
    if type == "unet":
        model = UNet11(pretrained=pretrained, num_classes=num_classes)
    elif type == "albunet":
        model = AlbuNet(pretrained=pretrained, num_classes=num_classes)
    elif type == "deeplab":
        model = deeplab.DeepLab(num_classes=num_classes, backbone='xception')
    else:
        raise Exception("Wrong model")
    return model


class Model(nn.Module):
    def __init__(self, pretrained=True, type="unet", num_classes=1):
        super().__init__()
        self.base = get_model(type, pretrained, num_classes)

    def forward(self, x):
        x = self.base(x)
        return x


class Model_boundary(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.base = UNet11_two_branch_out(pretrained=pretrained)

    def forward(self, x):
        y1, y2 = self.base(x)
        return y1, y2


def init_weights(module):
    if type(module) == nn.Linear:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)


class Model_comp(nn.Module):
    """model to compare two frames with image patch
    """

    def __init__(self, size=(224, 224), drop_rate=0.3, num=2):
        super().__init__()
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)

        # create resnet block
        resnet = models.resnet18(pretrained=True)
        self.res_extractor = nn.Sequential(
            *list(resnet.children())[:-1]
        )  # feature extractor with resnet

        # linear layer
        self.linear = nn.Sequential(
            nn.Linear(num * 512, 128),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(32, 1),
        )

        self.linear.apply(init_weights)

    def forward(self, x):
        # x shape: (B, 2, 3, 224, 224)
        B, num, _, h, w = x.shape
        x = x.view(B * num, 3, h, w)

        x_feat = self.res_extractor(x)
        x_feat = x_feat.view(B, num * 512)

        y = self.linear(x_feat)  # (B, 1)

        return y

