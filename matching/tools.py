import torchvision
from torch import nn
from torch.nn import functional as F
import torch
from torchvision import transforms
from torchvision import models
import numpy as np
import skimage
import cv2
import os
import sys
import models as custom_models


class Corr(nn.Module):
    def __init__(self, out_channel=100):
        super().__init__()

    def forward(self, x1, x2, corr_only=False):
        B, C, h1, w1 = x1.shape
        B, C, h2, w2 = x2.shape
        x_corr = torch.bmm(
            x1.permute(0, 2, 3, 1).view(B, h1 * w1, C),
            x1.view(B, C, -1)
        ) / C  # B, h1*w1, h2*w2
        x_b = x_corr.reshape(B*h1*w1, 1, h2, w2)
        # x_b = self.bn(x_b)
        if corr_only:
            x_b = x_b.reshape(B, h1*w1, h2*w2)
            return x_b
        out1 = x_b.reshape(B, h1 * w1, h2 * w2)
        out2 = x_b.reshape(B, h1 * w1, h2 * w2).transpose(-1, -2)

        out1, _ = torch.sort(out1, dim=-1)
        out2, _ = torch.sort(out2, dim=-1)

        out1 = F.adaptive_max_pool1d(out1, self.k)
        out2 = F.adaptive_max_pool1d(out2, self.k)

        out1 = out1.reshape(B, h1, w1, -1).permute(0, 3, 1, 2)
        out2 = out2.reshape(B, h2, w2, -1).permute(0, 3, 1, 2)

        return out1, out2




def std_mean(x):
    return (x-x.mean(dim=-3, keepdim=True))/(1e-8+x.std(dim=-3, keepdim=True))


class BusterModel(nn.Module):
    def __init__(self, topk=256):
        super().__init__()
        self.encoder = Encoder()  # out channel 512

        self.corrLayer = CrossCorr(out_channel=topk)
        self.bn1 = nn.BatchNorm2d(topk)
        self.bn2 = nn.BatchNorm2d(topk)

        self.decoder = Decoder(in_channel=topk)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.encoder(x1)
        x1 = std_mean(x1)
        x2 = self.encoder(x2)
        x2 = std_mean(x2)
        x_corr1, x_corr2 = self.corrLayer(x1, x2)

        xc1 = self.bn1(x_corr1)
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



class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=3, type='vgg'):
        super().__init__()

        if type == 'vgg':
            features = models.vgg19(pretrained=True).features

            self.feat1 = nn.Sequential(*list(features)[:2])
            self.feat2 = nn.Sequential(
                *list(features)[2:5]
            )
            self.feat3 = nn.Sequential(
                nn.Conv2d(64, 64, 1, 1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2, padding=0),
                nn.Dropout(0.3)
            )

            self.feat3.apply(weights_init_normal)
        # else:
        #     features = list(models.resnet101(pretrained=True).children())
        #     self.feat1 = nn.Sequential(*features[:3])
        #     self.feat2 = nn.Sequential(
        #         features[3],
        #         features[4]
        #     )

    def forward(self, x):
        x1 = self.feat1(x)  # 64, H, W
        x2 = self.feat2(x1)  # 64, H / 2, W / 2
        # x2 = F.interpolate(x2, size=x1.size()[-2:], mode='bicubic',
        #                    align_corners=True)
        # x3 = torch.cat((x1, x2), dim=-3)
        out = self.feat3(x2)  # 192, H/2, W/2
        return out, x1  # x2 is low-level features


class Normalizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        B, C, h1, w1 = x1.shape
        B, C, h2, w2 = x2.shape

        x1 = x1.view(B, C, h1 * w1)
        x2 = x2.view(B, C, h2 * w2)
        x_c = torch.cat([x1, x2], dim=-1)
        mean = x_c.mean(dim=-1, keepdim=True)
        std = x_c.mean(dim=-1, keepdim=True)

        x1 = (x1 - mean) / (std + 1e-8)
        x2 = (x2 - mean) / (std + 1e-8)

        x1 = x1.view(B, C, h1, w1)
        x2 = x2.view(B, C, h2, w2)
        # x1 = x1 / (1e-8 + torch.norm(x1, p=2, dim=-3, keepdim=True))
        # x2 = x2 / (1e-8 + torch.norm(x2, p=2, dim=-3, keepdim=True))

        return x1, x2


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.kaiming_normal_(m.weight.data)
        # torch.nn.init.normal_(m.weight.data, 1.0/3, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class MatcherPair(nn.Module):
    def __init__(self, in_channel=3, type='vgg', patch_size=(16, 16)):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channel=in_channel,
                                                  type=type)
        self.normalizer = Normalizer()
        # self.coef_ref = nn.Parameter(torch.tensor(10, dtype=torch.float32))
        # self.coef_temp = nn.Parameter(torch.tensor(10, dtype=torch.float32))

        self.tem_pool = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveMaxPool2d(patch_size)
        )

        self.k = 10
        self.n_div = (4, 4)

        self.final = nn.Sequential(
            nn.Conv2d(32+64, 1, 1),
        )

        self.out_conv1 = nn.Sequential(
            nn.Conv2d(self.n_div[0]*self.n_div[1], 32,
                      kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.out_conv2 = nn.Sequential(
            nn.ConvTranspose2d(32+64, 32, kernel_size=4, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.match_bnorm = nn.BatchNorm2d(1)

        self.final.apply(weights_init_normal)
        self.out_conv1.apply(weights_init_normal)
        self.out_conv2.apply(weights_init_normal)
        self.tem_pool.apply(weights_init_normal)

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)

    def corr(self, X, x):
        B, C, H, W = X.shape
        h, w = x.shape[-2:]
        X = X.reshape(1, B*C, H, W)

        hh = h//self.n_div[0]
        ww = w//self.n_div[1]
        x = x.reshape(B, C, self.n_div[0], hh, self.n_div[1], ww)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, hh, ww)

        match_map = F.conv2d(
            X, x, groups=B,
            padding=(hh//2, ww//2))  # 1, B * n^2, H, W
        match_map = F.interpolate(match_map, size=(H, W), mode='bilinear')
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_bnorm(match_map)
        match_map = match_map.squeeze(0).view(B, -1, H, W)
        return match_map

    def forward(self, x_ref, x_template):
        fref_feat, fref_low = self.feature_extractor(x_ref)  # C, 112, 112
        ftem_feat, _ = self.feature_extractor(x_template)  # C, 112, 112

        conf = self.corr(fref_feat, ftem_feat)

        out1 = self.out_conv1(conf)  # B, 32, 112, 112
        out_cat = torch.cat((out1, fref_feat), dim=-3)  # B, 32+64, 112, 112
        out = self.out_conv2(out_cat)  # B, 32, 224, 224
        out = F.interpolate(out, size=fref_low.shape[-2:], mode='bilinear',
                            align_corners=True)
        out = torch.cat((out, fref_low), dim=-3)  # B, 32+64, 224, 224
        # values = torch.mean(conf_values, dim=-1, keepdim=True)
        # values = values.reshape(-1, m, n, 1)
        out = self.final(out)
        return out


def IoU(r1, r2):
    x11, y11, w1, h1 = r1
    x21, y21, w2, h2 = r2
    x12 = x11 + w1
    y12 = y11 + h1
    x22 = x21 + w2
    y22 = y21 + h2
    x_overlap = max(0, min(x12, x22) - max(x11, x21))
    y_overlap = max(0, min(y12, y22) - max(y11, y21))
    I = 1. * x_overlap * y_overlap
    U = (y12-y11)*(x12-x11) + (y22-y21)*(x22-x21) - I
    J = I/U
    return J


def iou_mask(mask1, mask2):

    intersection = np.sum(np.logical_and(mask1, mask2), axis=(-1, -2))
    union = np.sum(np.logical_or(mask1, mask2), axis=(-1, -2))
    iou = intersection / (union + 1e-8)
    iou = iou[union > 1e-5]
    if iou.size == 0:
        val = 1
    else:
        val = np.mean(iou)
    return val


def iou_mask_with_ignore(mask_pred, mask_gt, thres=0.5):
    ignore_mask = (mask_gt > 0.45) & (mask_gt < 0.55)

    intersection = np.sum((mask_pred > thres) & (mask_gt > 0.5) & ~ignore_mask,
                          axis=(-1, -2))
    union = np.sum((mask_gt > 0.5) | (mask_pred > thres) & ~ignore_mask,
                   axis=(-1, -2))
    iou = intersection / (union + 1e-8)
    iou = iou[union > 1e-5]
    if iou.size == 0:
        val = 1
    else:
        val = np.mean(iou)
    return val


def evaluate_iou(rect_gt, rect_pred):
    # score of iou
    score = [IoU(i, j) for i, j in zip(rect_gt, rect_pred)]
    return score


def get_bbox(x):
    x = x.squeeze()
    r, c = np.nonzero(x)
    if r.size == 0:
        return None
    r1, r2 = np.min(r), np.max(r)
    c1, c2 = np.min(c), np.max(c)
    w, h = c2 - c1 + 1, r2 - r1 + 1
    return (c1, r1, w, h)


def locate_bbox(a, w, h):
    row = np.argmax(np.max(a, axis=1))
    col = np.argmax(np.max(a, axis=0))
    x = col - 1. * w / 2
    y = row - 1. * h / 2
    return x, y, w, h


def draw_rect(im, bbox, bbox2=None):
    x, y, w, h = [int(i) for i in bbox]
    cv2.rectangle(im, (x, y), (x+w, y+h), (0, 0, 1), 2)
    if bbox2 is not None:
        x, y, w, h = [int(i) for i in bbox2]
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 1, 0), 2)
    return im


def conv3x3(in_, out):
    return nn.Conv2d(in_, out, 3, padding=1)


class ConvRelu(nn.Module):
    def __init__(self, in_, out):
        super().__init__()
        self.conv = conv3x3(in_, out)
        self.activation = nn.ReLU(inplace=True)
        self.conv.apply(weights_init_normal)

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

    def forward(self, x):
        return self.block(x)


class CrossCorrV2(nn.Module):
    def __init__(self, ndiv=(10, 10), out_channel=100):
        super().__init__()
        self.n_div = ndiv
        # self.match_bnorm = nn.BatchNorm2d(1)
        self.out_channel = out_channel

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        h, w = x2.shape[-2:]
        x1 = x1 / torch.norm(x1, p=2, dim=1, keepdim=True)
        x2 = x2 / torch.norm(x2, p=2, dim=1, keepdim=True)

        x1 = x1.reshape(1, B*C, H, W)

        hh = h//self.n_div[0]
        ww = w//self.n_div[1]
        x2 = x2.reshape(B, C, self.n_div[0], hh, self.n_div[1], ww)
        x2 = x2.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, hh, ww)

        match_map = F.conv2d(
            x1, x2, groups=B,
            padding=(hh//2, ww//2)) / (hh * ww)  # 1, B * n^2, H, W
        match_map = F.interpolate(match_map, size=(H, W), mode='bilinear')
        match_map = match_map.permute(1, 0, 2, 3)
        # match_map = self.match_bnorm(match_map)
        match_map = match_map.squeeze(0).view(B, -1, H, W)
        out, _ = torch.topk(match_map, k=self.out_channel, dim=-3)
        out = torch.mean(out, dim=-3, keepdim=True)
        # out = out.view(B, 1, -1)
        # out = torch.softmax(out, dim=-1)
        # out = out.view(B, 1, H, W)
        return out


class MatchDeepLab(nn.Module):
    def __init__(self, im_size=320):
        super().__init__()
        self.im_size = im_size

        base = models.segmentation.deeplabv3_resnet101(pretrained=True)
        # self.backbone = base.backbone
        self.backbone = custom_models.CustomFeatureExtractor()

        # self.aspp = base.classifier[0]
        self.aspp = models.segmentation.deeplabv3.ASPP(256, [12, 24, 36])

        self.corr1 = CrossCorrV2(ndiv=(5, 5), out_channel=2)
        self.corr2 = CrossCorrV2(ndiv=(10, 10), out_channel=5)
        self.corr3 = CrossCorrV2(ndiv=(20, 20), out_channel=10)

        # self.head = models.segmentation.deeplabv3.DeepLabHead(
        #     100, num_classes=1)
        # self.head = models.segmentation.deeplabv3.ASPP(256, [12, 24, 36])
        self.final = nn.Sequential(
            base.classifier[1], base.classifier[2], base.classifier[3],
            nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))

        )

    def forward(self, x1, x2):
        feat1, flow1 = self.backbone(x1)['out']
        feat2, flow2 = self.backbone(x2)['out']

        feat1 = self.aspp(feat1)
        feat2 = self.aspp(feat2)

        corr1 = self.corr1(feat1, feat2)
        corr2 = self.corr2(feat1, feat2)
        corr3 = self.corr3(feat1, feat2)
        corr = corr1 * corr2 * corr3

        corr_low = self.corr3(F.interpolate(flow1, size=feat1.shape[-2:], mode='bicubic'),
                              F.interpolate(flow2, size=feat2.shape[-2:], mode='bicubic'))
        corr = corr * corr_low

        if torch.any(torch.isnan(feat1)):
            import pdb
            pdb.set_trace()

        out_h = feat1 * corr
        out = self.final(out_h)
        out = F.interpolate(out, size=self.im_size, mode='bilinear')
        # out2 = F.interpolate(out2, size=self.im_size, mode='bilinear')
        return out

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.final, self.aspp]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class MatchDeepLabV3p(nn.Module):
    def __init__(self, im_size=320):
        super().__init__()
        self.im_size = im_size
        encoder = models.vgg16_bn(
            pretrained=True).features[:24]  # 256, H/8, W/8
        self.low_feat = encoder[:7]
        self.high_feat = encoder[7:]
        self.normalizer = Normalizer()

        out_channel_corr = 10
        self.corr = CrossCorrV2(out_channel=out_channel_corr, ndiv=(5, 5))
        # self.head = models.segmentation.deeplabv3.DeepLabHead(
        #     100, num_classes=1)
        # self.head1 = models.segmentation.deeplabv3.ASPP(in_channels=100,
        #                                                 atrous_rates=[6, 12, 18])
        self.head1 = models.segmentation.deeplabv3.ASPP(in_channels=256,
                                                        atrous_rates=[12, 24, 36])
        self.low_conv = nn.Sequential(
            nn.Conv2d(256, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.head2 = nn.Sequential(
            nn.Conv2d(100+64, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(256),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(256),
            nn.Conv2d(256, 1, 1)
        )

        # self.head1.apply(weights_init_normal)
        self.head2.apply(weights_init_normal)
        self.low_conv.apply(weights_init_normal)
        self.corr.apply(weights_init_normal)

    def forward(self, x1, x2, corr_only=False):
        low_feat1 = self.low_feat(x1)
        low_feat2 = self.low_feat(x2)
        feat1 = self.high_feat(low_feat1)
        feat2 = self.high_feat(low_feat2)

        # feat1, feat2 = self.normalizer(feat1, feat2)
        # xcorr1, xcorr2 = self.corr(feat1, feat2)

        # x_low1 = self.low_conv(low_feat1)
        # x_low2 = self.low_conv(low_feat2)

        # x_aspp1 = self.head1(xcorr1)
        # x_aspp2 = self.head1(xcorr2)

        x_aspp1 = self.head1(feat1)
        x_aspp2 = self.head1(feat2)

        if corr_only:
            Corr = self.corr(x_aspp1, x_aspp2, corr_only=True)
            return Corr

        # xcorr1, xcorr2 = self.corr(x_aspp1, x_aspp2)
        xcorr1 = self.corr(x_aspp1, x_aspp2)

        x_low1 = self.low_conv(feat1)
        # x_low2 = self.low_conv(feat2)

        x1 = torch.cat((F.interpolate(xcorr1, size=x_low1.shape[-2:]),
                        x_low1), dim=1)
        # x2 = torch.cat((F.interpolate(xcorr2, size=x_low2.shape[-2:]),
        #                 x_low2), dim=1)

        out1 = self.head2(x1)
        # out2 = self.head2(x2)
        out1 = F.interpolate(out1, size=self.im_size, mode='bilinear')
        # out2 = F.interpolate(out2, size=self.im_size, mode='bilinear')
        return out1  # , out2

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)

    def get_1x_lr_params(self):
        modules = [self.low_feat, self.high_feat]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.corr, self.head1, self.head2, self.low_conv]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


class MatchUnet(nn.Module):
    def __init__(self, num_filters=32, pretrained=True, num_classes=1,
                 in_channels=3, im_size=320):
        """
        :param num_classes:
        :param num_filters:
        :param pretrained:
            False - no pre-trained network is used
            True  - encoder is pre-trained with VGG11
        """
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

        self.encoder = models.vgg11(pretrained=pretrained).features
        self.relu = self.encoder[1]
        if in_channels != 3:
            enc = self.encoder[0]
            self.conv1 = torch.nn.Conv2d(
                in_channels, enc.out_channels, kernel_size=enc.kernel_size,
                stride=enc.stride, padding=enc.padding)
            self.conv1.weight.data[:, :3].copy_(enc.weight.data[:, :3])
            self.conv1.weight.data[:, 3:].copy_(
                torch.cat([enc.weight.data[:, -1]]*(in_channels-3), dim=1))
            self.conv1.bias.data = enc.bias.data
        else:
            self.conv1 = self.encoder[0]
        self.conv2 = self.encoder[3]
        self.conv3s = self.encoder[6]
        self.conv3 = self.encoder[8]
        self.conv4s = self.encoder[11]
        self.conv4 = self.encoder[13]
        self.conv5s = self.encoder[16]
        self.conv5 = self.encoder[18]

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

        self.pre_center = nn.Conv2d((im_size//num_filters*2)**2+num_filters * 8 * 2,
                                    num_filters * 8 * 2, kernel_size=1)
        self.normalizer = Normalizer()
        self.pre_center.apply(weights_init_normal)
        self.final.apply(weights_init_normal)

        self.match_bnorm = nn.BatchNorm2d(1)
        # self.k = 5

    def encode(self, x):
        conv1 = self.relu(self.conv1(x))
        conv2 = self.relu(self.conv2(self.pool(conv1)))
        conv3s = self.relu(self.conv3s(self.pool(conv2)))
        conv3 = self.relu(self.conv3(conv3s))
        conv4s = self.relu(self.conv4s(self.pool(conv3)))
        conv4 = self.relu(self.conv4(conv4s))
        conv5s = self.relu(self.conv5s(self.pool(conv4)))
        conv5 = self.relu(self.conv5(conv5s))
        return conv5, conv4, conv3, conv2, conv1

    def decode(self, x, x_encode):
        conv5, conv4, conv3, conv2, conv1 = x_encode
        x = torch.cat((x, conv5), dim=-3)
        x = self.relu(self.pre_center(x))
        center = self.center(self.pool(x))

        dec5 = self.dec5(torch.cat([center, conv5], 1))
        dec4 = self.dec4(torch.cat([dec5, conv4], 1))
        dec3 = self.dec3(torch.cat([dec4, conv3], 1))
        dec2 = self.dec2(torch.cat([dec3, conv2], 1))
        dec1 = self.dec1(torch.cat([dec2, conv1], 1))
        return self.final(dec1)

    def normalize(self, x):
        return x / torch.norm(x, p=2, dim=-3, keepdim=True)

    def forward(self, x1, x2, corr_only=False):
        enc1 = self.encode(x1)
        enc2 = self.encode(x2)

        feat1, feat2 = enc1[0], enc2[0]

        # f1_norm = self.normalize(feat1)
        # f2_norm = self.normalize(feat2)
        # dist = torch.einsum('bcmn, bcpq -> bmnpq', (f1_norm, f2_norm))
        # _, m, n, p, q = dist.shape
        # # dist = dist.reshape(-1, m*n, p*q)

        # # conf_values, _ = torch.topk(dist, k=self.k, dim=-1)
        # # conf = conf_values.view(-1, m, n, self.k).permute(0, 3, 1, 2)

        # D1 = dist.reshape(-1, m, n, p*q).permute(0, 3, 1, 2)
        # D2 = dist.reshape(_, m*n, p, q)

        D = self.corr(feat1, feat2)  # B, h2, w2, h1, w1

        B, h2, w2, h1, w1 = D.shape
        D1 = D.view(B, h2 * w2, h1, w1)
        D2 = D.view(B, h2, w2, h1 * w1).permute(0, 3, 1, 2)

        if corr_only:
            return D1.reshape(B, h2*w2, -1).permute(0, 2, 1)

        out1 = self.decode(D1, enc1)
        out2 = self.decode(D2, enc2)

        return out1, out2

    def corr(self, X, x):
        B, C, H, W = X.shape
        h, w = x.shape[-2:]
        X = X.reshape(1, B*C, H, W)

        x = x.reshape(B, C, h, 1, w, 1)
        x = x.permute(0, 2, 4, 1, 3, 5).reshape(-1, C, 1, 1)

        match_map = F.conv2d(
            X, x, groups=B)  # 1, B * h * w, H, W
        #match_map = F.interpolate(match_map, size=(H, W), mode='bilinear')
        match_map = match_map.permute(1, 0, 2, 3)
        match_map = self.match_bnorm(match_map)
        match_map = match_map.squeeze(0).view(B, H, W, H, W)
        return match_map

    def set_bn_to_eval(self):
        def fn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.apply(fn)
