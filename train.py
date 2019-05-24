from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable


def BCE_loss(y, labels):
    y = y.squeeze().double()
    labels = labels.squeeze().double()

    wgt = torch.sum(labels) / (labels.shape[0] * labels.shape[1] * labels.shape[2])

    _loss = -(1 - wgt) * labels * F.logsigmoid(y) - \
        wgt * (1 - labels) * torch.log(1 - torch.sigmoid(y))
    _loss = torch.mean(_loss)

    if torch.isnan(_loss):
        import pdb; pdb.set_trace()

    return _loss.float()


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

