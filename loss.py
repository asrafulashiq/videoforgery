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


def BCE_loss_with_ignore(y, labels, with_weight=False, with_logits=True):
    eps = 1e-8
    y = y.contiguous().view(-1)
    labels = labels.contiguous().view(-1)

    _w = torch.sum(labels) / (labels.shape[0])

    if not with_weight:
        wgt = torch.ones_like(labels)
    else:
        wgt = labels * (1 - _w) + _w * (1 - labels)
    wgt[(labels > 0.4) & (labels < 0.6)] = 0

    if with_logits:
        bce_loss = F.binary_cross_entropy_with_logits(y, labels, wgt)
    else:
        bce_loss = F.binary_cross_entropy(y, labels, wgt)

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


def tri_loss_max(y, labels, r=0.05):

    B, C, H, W = y.shape

    lab_1 = (labels > 0.95).type_as(y)
    lab_2 = (labels <= 0.95).type_as(y)

    n_lab_1 = torch.norm(lab_1, p=0, dim=[-1, -2], keepdim=True)
    min_n = torch.min(n_lab_1)

    y_1 = y * lab_1
    y_2 = y * lab_2

    k = torch.max(torch.ceil((min_n * r)).int(), torch.tensor(1, dtype=torch.int).cuda())
    top1 = torch.mean(torch.topk(y_1.view(B, C, -1), k=k, dim=-1)[0], dim=-1, keepdim=True)
    top2 = torch.mean(torch.topk(y_2.view(B, C, -1), k=k, dim=-1)[0], dim=-1, keepdim=True)

    delta = 0.5

    loss = torch.max(
        delta - (top1 - top2), torch.tensor(0., dtype=top1.dtype).cuda()
    ) * (n_lab_1 > 0).type_as(top1)

    loss = loss.mean()

    if torch.isnan(loss):
        import pdb; pdb.set_trace()

    return loss
