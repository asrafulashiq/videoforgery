from torch import nn
from torch.nn import functional as F
import torch
from torchvision import models
import torchvision


def init_weights(module):
    classname = module.__class__.__name__
    if classname.find('Linear') != -1 or classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.0)

def conv3x3(in_, out, kernel=3):
    return nn.Conv2d(in_, out, kernel, padding=kernel//2)


class ConvRelu(nn.Module):
    def __init__(self, in_, out, kernel=3):
        super().__init__()
        self.conv = conv3x3(in_, out, kernel=kernel)
        self.activation = nn.ReLU(inplace=True)

        self.apply(init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            ConvRelu(in_channels, middle_channels),
            nn.ConvTranspose2d(
                middle_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            ),
            nn.ReLU(inplace=True),
        )

        self.apply(init_weights)

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        encoder = models.vgg11(pretrained=pretrained).features

        self.relu = encoder[1]

        self.conv1 = encoder[0]
        self.conv2 = encoder[3]
        self.conv3s = encoder[6]
        self.conv3 = encoder[8]
        self.conv4s = encoder[11]
        self.conv4 = encoder[13]
        self.conv5s = encoder[16]
        self.conv5 = encoder[18]

    def forward(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))

        intermediate_conv = [
            conv1, conv2, conv3, conv4, conv5
        ]

        return intermediate_conv


class Decoder(nn.Module):
    def __init__(self, num_filters=32, num_classes=1):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)

        self.center = DecoderBlock(
            num_filters * 8 * 2, num_filters * 8 * 2, num_filters * 8
        )

        self.dec5 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 8
        )
        self.dec4 = DecoderBlock(
            num_filters * (16 + 8), num_filters * 8 * 2, num_filters * 4
        )
        self.dec3 = DecoderBlock(
            num_filters * (8 + 4), num_filters * 4 * 2, num_filters * 2
        )
        self.dec2 = DecoderBlock(
            num_filters * (4 + 2), num_filters * 2 * 2, num_filters
        )
        self.dec1 = ConvRelu(num_filters * (2 + 1), num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

        self.final.apply(init_weights)

    def forward(self, x, convs):

        conv1, conv2, conv3, conv4, conv5 = convs
        center = self.center(self.pool(conv5))
        # first branch
        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        out = self.final(dec1)

        return out


class Center(nn.Module):
    def __init__(self, in_features=256, span_len=4):
        super().__init__()
        self.center_reduce = ConvRelu(in_features, in_features//span_len,
                                      kernel=1)

    def forward(self, x):
        return self.center_reduce(x)


class TemporalBlock(nn.Module):
    def __init__(self, in_channel, span_kernel=2):
        super().__init__()
        self.span_kernel = span_kernel
        self.conv = ConvRelu(in_channel*2, in_channel,
                             kernel=3)
        self.stack = Stack()

    def forward(self, x):
        # x: L, C, H, W
        # L, C, H, W = x.shape
        x = self.stack(x, n_stack=self.span_kernel-1)
        x = self.conv(x)
        return x

class TemporalNet(nn.Module):
    def __init__(self, in_channel, level=1):
        super().__init__()
        layers = []
        for i in range(level):
            span_len = 2**(i+1)
            layers.append(
                TemporalBlock(in_channel, span_len)
            )
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class Stack(nn.Module):
    def __init__(self, n_stack=1):
        super().__init__()

    def forward(self, x, n_stack=1):
        # x : L, C, H, W
        x_pad = F.pad(x, (0,0, 0,0, 0,0, n_stack, 0), mode='constant', value=0)
        # new shape: (L+ns), C, H, W
        x_stack = torch.cat((x_pad[0:-n_stack], x_pad[n_stack:]), dim=1)
        return x_stack


class TCN(nn.Module):
    def __init__(self, span_len=2, num_filters=32, pretrained=True):
        super().__init__()
        self.encoder = Encoder(pretrained)
        self.center_reduce = ConvRelu(num_filters*16, 64, kernel=1)
        self.temp = TemporalNet(64, level=1)
        self.center_expand = ConvRelu(64,
                                      num_filters*16,
                                      kernel=1)

        self.decoder = Decoder(num_filters=num_filters, num_classes=1)

    def forward(self, x):
        # x: L, 3, H, W
        x_encoder_convs = self.encoder(x)  # out last: L, 256, 14, 14
        conv5 = x_encoder_convs[-1]

        x_center = self.center_reduce(conv5)  # out: L, 64, 14, 14
        center_tcn = self.temp(x_center)  # out: L, 64, 14, 14
        x_expand = self.center_expand(center_tcn)  # out: L, 256, 14, 14
        # x_expand = conv5

        out = self.decoder(x_expand, x_encoder_convs)

        return out
