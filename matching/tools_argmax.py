import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from matching.argmax import SoftArgmax2D
import numpy as np


class Corr(nn.Module):
    def __init__(self, hw=(60, 60)):
        super().__init__()
        self.h = hw[0]
        self.w = hw[1]
        self.argmax = SoftArgmax2D(window_fn="Parzen", window_width=10)

        ind_arr = np.flip(np.indices(hw), 0).astype(np.float32)
        ind_arr[0] = ind_arr[0] / self.w
        ind_arr[1] = ind_arr[1] / self.h

        self.ind_arr = torch.tensor(ind_arr.copy(), dtype=torch.float).cuda()

    def forward(self, x1, x2):

        b, c, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape

        x_c = torch.matmul(x1.permute(0, 2, 3, 1).view(b, -1, c),
                           x2.view(b, c, -1)) / c  # h1 * w1, h2 * w2
        # x_c = F.softmax(x_c, dim=-1) * F.softmax(x_c, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc1 = x_c.view(b, h1*w1, h2, w2)
        ind_max = self.argmax(xc1)
        ind_max[..., 0] = ind_max[..., 0] / w2
        ind_max[..., 1] = ind_max[..., 1] / h2
        ind_max = ind_max.view(b, h1, w1, -1).permute(0, 3, 1, 2)

        ind_max = ind_max - self.ind_arr

        return ind_max


def std_mean(x):
    return (x-x.mean(dim=-3, keepdim=True))/(1e-8+x.std(dim=-3, keepdim=True))


class BusterModel(nn.Module):
    def __init__(self, hw=(20, 20)):
        super().__init__()
        self.encoder = Extractor_VGG19()

        self.corrLayer = Corr(hw=hw)

        self.decoder = Decoder(in_channel=2)

        self.hw = hw

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.encoder(x1, out_size=self.hw)
        x1 = std_mean(x1)
        x2 = self.encoder(x2, out_size=self.hw)
        x2 = std_mean(x2)

        xc1 = self.corrLayer(x1, x2)

        # xc1 = self.bn1(x_corr1)
        # xc2 = self.bn2(x_corr2)

        out1 = self.decoder(xc1)
        # out2 = self.decoder(xc2)
        out1 = F.interpolate(out1, size=(h, w), mode='bilinear')
        return out1

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)

    def get_1x_lr_params(self):
        modules = [self.encoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.decoder, self.corrLayer]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class CustomConv(nn.Module):
    def __init__(self, in_channel=64, out_channel=64, kernel_size=3,
                 activation='relu', num=1, pool=None):
        super().__init__()
        layers = []
        first = True
        for i in range(num):
            if first:
                in_c = in_channel
                first = False
            else:
                in_c = out_channel
            layers.append(
                nn.Conv2d(in_c, out_channel, kernel_size,
                          padding=kernel_size//2)
            )
            if activation is not None:
                layers.append(nn.ReLU())
        if pool is not None:
            layers.append(nn.MaxPool2d(2, stride=2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_channel=3):
        super().__init__()

        self.layer1 = CustomConv(in_channel, 64, num=2, pool=True)
        self.layer2 = CustomConv(64, 128, num=2, pool=True)
        self.layer3 = CustomConv(128, 256, num=3, pool=True)
        self.layer4 = CustomConv(256, 512, num=3, pool=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


class BNInception(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, filt_list=[1, 3, 5]):
        super().__init__()
        layers = []
        for filt in filt_list:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size=filt,
                          padding=filt//2)
            )
        self.layers = nn.ModuleList(layers)
        self.bn = nn.BatchNorm2d(len(filt_list) * out_channel)
        self.relu = nn.ReLU()

    def forward(self, x):
        uc = []
        for conv in self.layers:
            uc.append(conv(x))
        x = torch.cat(uc, dim=-3)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channel=64):
        super().__init__()

        self.binc1 = BNInception(in_channel, 8)
        self.binc2 = BNInception(24, 6)
        self.binc3 = BNInception(42, 4)
        self.binc4 = BNInception(54, 2)
        self.binc5 = BNInception(60, 2)
        self.binc6 = BNInception(66, 2, filt_list=[5, 7, 11])

        self.last_conv = nn.Conv2d(6, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.binc1(x)
        x = F.interpolate(x, scale_factor=2, mode='bilinear')

        x1 = self.binc2(x)
        x = torch.cat((
            F.interpolate(x, scale_factor=2, mode='bilinear'),
            F.interpolate(x1, scale_factor=2, mode='bilinear')
        ), dim=-3)

        x1 = self.binc3(x)
        x = torch.cat((
            F.interpolate(x, scale_factor=2, mode='bilinear'),
            F.interpolate(x1, scale_factor=2, mode='bilinear')
        ), dim=-3)

        x1 = self.binc4(x)
        x = torch.cat((
            F.interpolate(x, scale_factor=2, mode='bilinear'),
            F.interpolate(x1, scale_factor=2, mode='bilinear')
        ), dim=-3)

        x1 = self.binc5(x)
        x = torch.cat((
            x, x1
        ), dim=-3)

        x = self.binc6(x)
        out = self.last_conv(x)

        return out


class FeatureExtractor(nn.Sequential):
    def __init__(self):
        super(FeatureExtractor, self).__init__()

    def forward(self, x, out_size=(40, 40)):
        outputs = []
        for module in self._modules:
            x = self._modules[module](x)
            outputs.append(F.interpolate(x, size=out_size, mode='bilinear',
            align_corners=True))
        out = torch.cat(outputs, dim=-3)
        return out


class Extractor_VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_temp = torchvision.models.vgg19(pretrained=True).features
        self.model = FeatureExtractor()

        conv_counter = 1
        relu_counter = 1
        block_counter = 1

        for i, layer in enumerate(list(cnn_temp)):

            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(block_counter) + "_" + \
                    str(conv_counter) + "__" + str(i)
                conv_counter += 1
                self.model.add_module(name, layer)

            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(block_counter) + "_" + \
                    str(relu_counter) + "__" + str(i)
                relu_counter += 1
                self.model.add_module(name, nn.ReLU(inplace=False))

            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(block_counter) + "__" + str(i)
                relu_counter = conv_counter = 1
                block_counter += 1
                self.model.add_module(name, nn.MaxPool2d((2, 2)))


    def forward_subnet(self, input_tensor, L):

        if L == 5:
            start_layer, end_layer = 21, 29  # From Conv4_2 to ReLU5_1 inclusively
        elif L == 4:
            start_layer, end_layer = 12, 20  # From Conv3_2 to ReLU4_1 inclusively
        elif L == 3:
            start_layer, end_layer = 7, 11  # From Conv2_2 to ReLU3_1 inclusively
        elif L == 2:
            start_layer, end_layer = 2, 6  # From Conv1_2 to ReLU2_1 inclusively
        else:
            raise ValueError("Invalid layer number")

        for i, layer in enumerate(list(self.model)):
            if i >= start_layer and i <= end_layer:
                input_tensor = layer(input_tensor)
        return input_tensor

    def forward(self, img_tensor, out_size=(40, 40)):
        features = self.model(img_tensor, out_size)
        return features
