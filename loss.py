from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(x, t, gamma=2, with_weight=False):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,1, ...].
        y: (tensor) sized [N, 1, ...].
    Return:
        (tensor) focal loss.
    '''

    x = x.view(-1)
    t = t.view(-1)

    if with_weight:
        wgt = torch.sum(t) / (t.shape[0])
    else:
        wgt = 0.5

    p = torch.sigmoid(x)
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = (1-wgt)*t + wgt*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach())


def dice_loss(y, labels):
    smooth = 1
    y = torch.sigmoid(y.view(-1))
    lab = labels.view(-1)

    numer = 2 * (y * lab).sum()
    den = y.sum() + lab.sum()

    return 1 - (numer + smooth) / (den + smooth)


def BCE_loss(y, labels, with_weight=False, with_logits=True):
    eps = 1e-8
    y = y.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    _w = torch.sum(labels) / (labels.shape[0])


    if not with_weight:
        wgt=None
    else:
        wgt = labels * (1 - _w) + _w * (1 - labels)

    if with_logits:
        bce_loss = F.binary_cross_entropy_with_logits(y, labels, wgt)
    else:
        bce_loss = F.binary_cross_entropy(y, labels, wgt)


    if torch.isnan(bce_loss) or bce_loss < 0:
        import pdb
        pdb.set_trace()

    return bce_loss.float()


def BCE_loss_with_src(y, labels, with_weight=False, with_logits=True):

    loss1 = BCE_loss(y[:, 0], labels[:, 0], with_weight=with_weight, with_logits=True)
    loss2 = BCE_loss(y[:, 1], labels[:, 1], with_weight=with_weight, with_logits=True)
    return loss1, loss2



def Two_loss(y, labels):
    y = torch.softmax(y, dim=1)
    lab_forge = (labels == 2).type_as(y)
    loss_forge = BCE_loss(y[:, 2], lab_forge, with_logits=False)

    lab_src = (labels==1).type_as(y)
    loss_src = torch.mean(-lab_src * torch.log(y[:, 1] + 1e-8 ))

    lab_back = (labels == 0).type_as(y)
    loss_back = torch.mean(-lab_back * torch.log(y[:, 0] + 1e-8))


    alpha = 0.5
    loss = loss_forge + loss_src + 0.5 * loss_back
    return loss


def CrossEntropy2d(input, target):
    # input:(n, c, h, w) target:(n, h, w)
    n, c, h, w = input.size()

    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0].view(-1, c)

    target_mask = target >= 0
    target = target[target_mask]

    # weight = torch.tensor([0.2, 0.3, 0.5]).cuda()
    loss = F.cross_entropy(input, target)

    return loss
