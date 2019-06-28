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
        # B, C, h1, w1 = x1.shape
        # B, C, h2, w2 = x2.shape

        # x1 = x1.view(B, C, h1 * w1)
        # x2 = x2.view(B, C, h2 * w2)
        # x_c = torch.cat([x1, x2], dim=-1)
        # mean = x_c.mean(dim=-1, keepdim=True)
        # std = x_c.mean(dim=-1, keepdim=True)

        # x1 = (x1 - mean) / (std + 1e-8)
        # x2 = (x2 - mean) / (std + 1e-8)

        # x1 = x1.view(B, C, h1, w1)
        # x2 = x2.view(B, C, h2, w2)
        x1 = x1 / (1e-8 + torch.norm(x1, p=2, dim=-3, keepdim=True))
        x2 = x2 / (1e-8 + torch.norm(x2, p=2, dim=-3, keepdim=True))

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

        # fref, ftem = self.normalizer(fref_feat, ftem_feat)

        # # downsample ftem to (12, 12)
        # ftem = self.tem_pool(ftem)

        # dist = torch.einsum('bcmn, bcpq -> bmnpq', (fref, ftem))
        # _, m, n, p, q = dist.shape
        # dist = dist.reshape(-1, m*n, p*q)

        # # conf_ref = F.softmax(self.coef_ref * dist, dim=-2)
        # # conf_tmp = F.softmax(self.coef_temp * dist, dim=-1)
        # # confidence = torch.sqrt(conf_ref * conf_tmp)

        # confidence = dist

        # conf_values, _ = torch.topk(confidence, k=self.k, dim=-1)
        # conf = conf_values.view(-1, m, n, self.k).permute(0, 3, 1, 2)
        # k, 112, 112

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
    val = np.mean(iou)
    return val


def iou_mask_with_ignore(mask_pred, mask_gt):
    ignore_mask = (mask_gt > 0.45) & (mask_gt < 0.55)

    intersection = np.sum((mask_pred > 0.5) & (mask_gt > 0.5) & ~ignore_mask,
                          axis=(-1, -2))
    union = np.sum((mask_gt > 0.5) | (mask_pred > 0.5) & ~ignore_mask,
                   axis=(-1, -2))
    iou = intersection / (union + 1e-8)
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

    def forward(self, x1, x2):
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
