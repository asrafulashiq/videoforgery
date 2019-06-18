from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11, AlbuNet, UNet11_two_branch_out
from torchvision import transforms
from torchvision import models
import numpy as np
from modeling import deeplab
# from unet.unet_model import UNet


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def get_model(type, pretrained=True, num_classes=1):
    if type == "unet":
        model = UNet11(pretrained=pretrained, num_classes=num_classes)
    elif type == "albunet":
        model = AlbuNet(pretrained=pretrained, num_classes=num_classes)
    elif type == "deeplab":
        model = deeplab.DeepLab(num_classes=num_classes, backbone='xception')
    elif type == "unet-basic":
        model = UNet11(pretrained=pretrained, num_classes=num_classes)
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
    classname = module.__class__.__name__
    if classname.find('Linear') != -1 or classname.find("Conv") != -1:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)


class Discriminator(nn.Module):
    def __init__(self, in_channels=4):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters,
                                4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

        self.apply(init_weights)

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)



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

