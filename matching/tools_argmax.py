import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
from torchvision import transforms
from matching.argmax import SoftArgmax2D
import numpy as np


class Corr(nn.Module):
    def __init__(self, hw=(60, 60)):
        super().__init__()
        self.h = hw[0]
        self.w = hw[1]
        self.argmax = SoftArgmax2D(window_fn="Parzen", window_width=10, wt=5)

        ind_arr = np.flip(np.indices(hw), 0).astype(np.float32)
        ind_arr[0] = ind_arr[0] / self.w
        ind_arr[1] = ind_arr[1] / self.h

        self.alpha = nn.Parameter(torch.tensor(10., dtype=torch.float32)).cuda()

        self.ind_arr = torch.tensor(ind_arr.copy(), dtype=torch.float).cuda()

    def forward(self, x1, x2):

        b, c, h1, w1 = x1.shape
        _, _, h2, w2 = x2.shape

        x_c_o = torch.matmul(x1.permute(0, 2, 3, 1).view(b, -1, c),
                           x2.view(b, c, -1))  # h1 * w1, h2 * w2
        x_c = F.softmax(x_c_o*self.alpha, dim=-1) * \
            F.softmax(x_c_o*self.alpha, dim=-2)
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        xc1 = x_c.view(b, h1*w1, h2, w2)
        ind_max = self.argmax(xc1)
        ind_max[..., 0] = ind_max[..., 0] / w2
        ind_max[..., 1] = ind_max[..., 1] / h2
        ind_max = ind_max.view(b, h1, w1, -1).permute(0, 3, 1, 2)

        ind_max1 = ind_max - self.ind_arr

        xc2 = x_c.view(b, h1, w1, h2 * w2).permute(0, 3, 1, 2).contiguous()
        ind_max = self.argmax(xc2)
        ind_max[..., 0] = ind_max[..., 0] / w2
        ind_max[..., 1] = ind_max[..., 1] / h2
        ind_max = ind_max.view(b, h1, w1, -1).permute(0, 3, 1, 2)

        ind_max2 = ind_max - self.ind_arr



        return ind_max1, ind_max2


def std_mean(x):
    return (x-x.mean(dim=-3, keepdim=True))/(1e-8+x.std(dim=-3, keepdim=True))

def plot(x, name='1'):
    if x.shape[0] == 2:
        x = torch.cat((x, x[[0]]), dim=-3)
    x = F.interpolate(x.unsqueeze(0), size=(120, 120), mode='bilinear').squeeze(0)
    fn = lambda x: (x-x.min())/(x.max()-x.min()+1e-8)
    torchvision.utils.save_image(fn(x), f'{name}.png')


class BusterModel(nn.Module):
    def __init__(self, hw=(40, 40)):
        super().__init__()
        self.hw = hw

        self.encoder = Extractor_VGG19()

        self.corrLayer = Corr(hw=hw)

        self.aspp = models.segmentation.deeplabv3.ASPP(in_channels=96,
                                                       atrous_rates=[6, 12, 24])

        self.low_conv = nn.Sequential(
            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.corr_conv = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(256),
            nn.Conv2d(256, 1, 1)
        )

        self.head.apply(weights_init_normal)
        self.low_conv.apply(weights_init_normal)
        self.corr_conv.apply(weights_init_normal)

    def forward(self, x1, x2):
        input1, input2 = x1, x2
        b, c, h, w = x1.shape
        x1, x1_low = self.encoder(x1, out_size=self.hw)
        # x1 = std_mean(x1)
        x1 = F.normalize(x1, p=2, dim=-3)
        x2, _ = self.encoder(x2, out_size=self.hw)
        x2 = F.normalize(x2, p=2, dim=-3)
        # x2 = std_mean(x2)

        xc1, xc2 = self.corrLayer(x1, x2)

        x1_low = self.low_conv(x1_low)
        x1_c = self.corr_conv(xc1)

        xcl = torch.cat((x1_low, x1_c), dim=-3)

        out = self.aspp(xcl)
        out = self.head(out)
        out = F.interpolate(out, size=(h, w), mode='bilinear')

        xc1 = F.interpolate(xc1, size=(h, w), mode='bilinear')
        xc2 = F.interpolate(xc2, size=(h, w), mode='bilinear')

        return out #, xc1, xc2

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
        modules = [self.aspp, self.corrLayer, self.low_conv, self.corr_conv,
                   self.head]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.normal_(m.weight.data, 1.0/3, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


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


class Extractor_VGG19(nn.Module):
    def __init__(self):
        super().__init__()
        cnn_temp = torchvision.models.vgg19(pretrained=True).features

        self.layer1 = nn.Sequential(cnn_temp[:10])  # stride 4
        self.layer2 = nn.Sequential(cnn_temp[10:19])  # s 8
        self.layer3 = nn.Sequential(cnn_temp[19:28])  # s 16

    def forward(self, x, out_size=(40, 40)):
        x1 = self.layer1(x)
        x1_u = F.interpolate(x1, size=out_size, mode='bilinear')

        x2 = self.layer2(x1)
        x2_u = F.interpolate(x2, size=out_size, mode='bilinear')

        x3 = self.layer3(x2)
        x3_u = F.interpolate(x3, size=out_size, mode='bilinear')

        x_all = torch.cat([x1_u, x2_u, x3_u], dim=-3)

        return x_all, x1_u
