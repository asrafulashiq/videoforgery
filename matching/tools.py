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
        x1 = x1 / (1e-8 + torch.norm(x1, p=2, dim=1, keepdim=True))
        x2 = x2 / (1e-8 + torch.norm(x2, p=2, dim=1, keepdim=True))

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
