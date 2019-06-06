from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def focal_loss(x, t, gamma=2):
    '''Focal loss.
    Args:
        x: (tensor) sized [N,1, ...].
        y: (tensor) sized [N, 1, ...].
    Return:
        (tensor) focal loss.
    '''

    x = x.view(-1)
    t = t.view(-1)

    wgt = torch.sum(t) / (t.shape[0])

    p = torch.sigmoid(x)
    pt = p*t + (1-p)*(1-t)         # pt = p if t > 0 else 1-p
    w = (1-wgt)*t + wgt*(1-t)  # w = alpha if t > 0 else 1-alpha
    w = w * (1-pt).pow(gamma)
    return F.binary_cross_entropy_with_logits(x, t, w.detach())


def BCE_loss(y, labels):
    eps = 1e-8
    y = y.squeeze().double()
    labels = labels.squeeze().double()

    wgt = torch.sum(labels) / (labels.shape[0] * labels.shape[1] * labels.shape[2])

    _loss = -(1 - wgt) * labels * F.logsigmoid(y) - wgt * (1 - labels) * torch.log(
        1 - torch.sigmoid(y) + eps
    )
    _loss = torch.mean(_loss)

    if torch.isnan(_loss) or _loss < 0:
        import pdb

        pdb.set_trace()

    return _loss.float()


def Index_loss(ysim, ydis, ind_gt, device):
    ysim = torch.pow(torch.sigmoid(ysim.squeeze()), 2)
    ydis = torch.pow(torch.sigmoid(ydis.squeeze()), 2)
    ind_gt = ind_gt.squeeze()

    y_gt = torch.abs(ind_gt[:, 0] - ind_gt[:, 1])
    # _loss = torch.mean(torch.max(y_gt - y, torch.FloatTensor([0]).to(device)))
    _loss = torch.mean(torch.max(0.2 + ysim - ydis, torch.FloatTensor([0]).to(device)))

    if torch.isnan(_loss):
        import pdb

        pdb.set_trace()

    return _loss


def train_match_in_the_video(
    dataset, model, optimizer, args, iteration, device, logger=None
):
    model.train()
    X_im, Ind = dataset.load_triplet(num=args.batch_size)
    X_im = X_im.to(device)
    Ind = Ind.to(device)

    y_sim = model(X_im[:, [0, 1]])
    y_dis = model(X_im[:, [0, 2]])

    loss = Index_loss(y_sim, y_dis, Ind, device)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(f"Iteration: {iteration}  Loss : {loss.data.cpu().numpy():.4f}")

    if logger is not None:
        logger.add_scalar("loss/loss_ind", loss, iteration)


def train(inputs, labels, model, optimizer, args, iteration, device, logger=None):

    model.train(True)
    inputs = inputs.to(device)
    labels = labels.to(device)

    # prediction
    y = model(inputs)

    # loss = focal_loss(y, labels)
    loss = BCE_loss(y, labels)

    optimizer.zero_grad()
    loss.backward()

    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    loss_val = loss.data.cpu().numpy()
    print(f"Iteration: {iteration}  Loss : {loss_val:.4f}")

    if logger is not None:
        logger.add_scalar("loss/total", loss, iteration)


def train_with_boundary(
    inputs, labels, model, optimizer, args, iteration, device, logger=None
):

    model.train(True)
    inputs = inputs.to(device)
    labels = labels.to(device)

    mask, boundary = labels[:, 0], labels[:, 1]
    # prediction
    y_m, y_b = model(inputs)

    loss_mask = BCE_loss(y_m, mask)
    loss_boundary = BCE_loss(y_b, boundary)

    loss = loss_mask + args.gamma_b * loss_boundary

    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    optimizer.step()

    loss_val = loss.data.cpu().numpy()

    print(
        f"Iteration: {iteration:6d}, loss_m : {loss_mask.data.cpu().numpy():.4f} "
        + f"loss_b : {loss_boundary.data.cpu().numpy():.4f} "
        + f"Loss : {loss_val:.4f}"
    )

    if logger is not None:
        logger.add_scalar("loss/total", loss, iteration)
        logger.add_scalar("loss/mask", loss_mask, iteration)
        logger.add_scalar("loss/boundary", loss_boundary, iteration)
