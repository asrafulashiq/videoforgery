from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def BCE_loss(y, labels):
    eps = 1e-8
    y = y.squeeze().double()
    labels = labels.squeeze().double()

    wgt = torch.sum(labels) / (labels.shape[0] * labels.shape[1] * labels.shape[2])

    _loss = -(1 - wgt) * labels * F.logsigmoid(y) - \
        wgt * (1 - labels) * torch.log(1 - torch.sigmoid(y) + eps)
    _loss = torch.mean(_loss)

    if torch.isnan(_loss):
        import pdb; pdb.set_trace()

    return _loss.float()

def Index_loss(y, ind_gt, device):
    y = y.squeeze()
    ind_gt = ind_gt.squeeze()

    y_gt = torch.abs(ind_gt[:, 0] - ind_gt[:, 1])

    # _loss = torch.mean(torch.max(y_gt - y, torch.FloatTensor([0]).to(device)))
    _loss = torch.mean(torch.abs(y_gt - y))

    if torch.isnan(_loss):
        import pdb; pdb.set_trace()

    return _loss



def train_match_in_the_video(dataset, model, optimizer,
                             args, iteration, device, logger=None):
    X_im, Ind = dataset.load_triplet(num=10)
    X_im = X_im.to(device)
    Ind = Ind.to(device)

    f_comp = model(X_im)

    loss = Index_loss(f_comp, Ind, device)

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

    loss = BCE_loss(y, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iteration: {iteration}  Loss : {loss.data.cpu().numpy():.4f}")

    if logger is not None:
        logger.add_scalar("loss/total", loss, iteration)

