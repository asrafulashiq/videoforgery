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
        wgt = None
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


def BCE_loss_with_ignore(y, labels, with_weight=False, with_logits=True):
    eps = 1e-8
    y = y.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    ignore_ind = ((labels > 0.4) & (labels < 0.6))

    ind_pos = (labels > 0.9) & ~ignore_ind
    ind_neg = (labels < 0.1) & ~ignore_ind
    ind_total = ind_pos.sum()+ind_neg.sum()

    _w = torch.max(
        (ind_pos.sum() / (ind_total + 1e-8)).float(),
        torch.tensor(0.05).to(y.device)
    )

    if not with_weight:
        wgt = torch.ones_like(labels)
    else:
        wgt = labels * (1 - _w) + _w * (1 - labels)
    wgt[ignore_ind] = 0

    if with_logits:
        bce_loss = F.binary_cross_entropy_with_logits(y, labels, wgt, reduction='none')
    else:
        bce_loss = F.binary_cross_entropy(y, labels, wgt, reduction='none')
    
    bce_loss = bce_loss.sum() / wgt.sum()

    if torch.isnan(bce_loss) or bce_loss < 0:
        import pdb
        pdb.set_trace()

    return bce_loss.float()


def dice_loss_with_ignore(y, labels):
    smooth = 10
    y = torch.sigmoid(y.view(-1))
    lab = labels.view(-1)

    wgt = torch.ones_like(lab)
    wgt[(lab > 0.4) & (lab < 0.6)] = 0

    numer = 2 * (y * lab * wgt).sum()
    den = (y * wgt).sum() + (lab * wgt).sum()

    return 1 - (numer + smooth) / (den + smooth)


def BCE_loss_with_src(y, labels, with_weight=False, with_logits=True):

    loss1 = BCE_loss(y[:, 0], labels[:, 0],
                     with_weight=with_weight, with_logits=True)
    loss2 = BCE_loss(y[:, 1], labels[:, 1],
                     with_weight=with_weight, with_logits=True)
    return loss1, loss2


# def sim_loss(y1, y2, m1, m2):
#     y2 = -y2

#     if torch.all(m2 < 0.8) or torch.all(m1 < 0.8):
#         return 0

#     val1 = y1 * (m1.expand_as(y1) > 0.8).type_as(y1)
#     val2 = y2 * (m2.expand_as(y2) > 0.8).type_as(y1)

#     val1.sum(dim)


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
