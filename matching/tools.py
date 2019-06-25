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
                *list(features)[2:18]
            )
        else:
            features = list(models.resnet101(pretrained=True).children())
            self.feat1 = nn.Sequential(*features[:3])
            self.feat2 = nn.Sequential(
                features[3],
                features[4]
            )

    def forward(self, x):
        x1 = self.feat1(x)
        x2 = self.feat2(x1)

        x2 = F.interpolate(x2, size=x1.size()[-2:], mode='bicubic',
                           align_corners=True)
        x_feat = torch.cat((x1, x2), dim=-3)
        return x_feat


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
        return x1, x2


class MatcherPair(nn.Module):
    def __init__(self, in_channel=3, type='vgg'):
        super().__init__()
        self.feature_extractor = FeatureExtractor(in_channel=in_channel,
                                                  type=type)
        self.normalizer = Normalizer()
        self.coef_ref = nn.Parameter(torch.tensor(1, dtype=torch.float32))
        self.coef_temp = nn.Parameter(torch.tensor(1, dtype=torch.float32))

    def forward(self, x_ref, x_template):
        fref = self.feature_extractor(x_ref)
        ftem = self.feature_extractor(x_template)

        fref, ftem = self.normalizer(fref, ftem)
        dist = torch.einsum('bcmn, bcpq -> bmnpq', (fref, ftem))
        _, m, n, p, q = dist.shape
        dist = dist.reshape(-1, m*n, p*q)
        conf_ref = F.softmax(self.coef_ref * dist, dim=-2)
        conf_tmp = F.softmax(self.coef_temp * dist, dim=-1)

        confidence = torch.sqrt(conf_ref * conf_tmp)

        conf_values, _ = torch.topk(confidence, k=10)

        values = torch.mean(conf_values, dim=-1, keepdim=True)
        values = values.reshape(-1, m, n, 1)
        return values


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


def evaluate_iou(rect_gt, rect_pred):
    # score of iou
    score = [IoU(i, j) for i, j in zip(rect_gt, rect_pred)]
    return score


def get_bbox(x):
    x = x.squeeze()
    r, c = np.nonzero(x)
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
