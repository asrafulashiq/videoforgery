from torch import nn
from torch.nn import functional as F
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

def BCE_loss(y, labels):
    _loss = F.binary_cross_entropy_with_logits(y, labels)
    return _loss


def train(inputs, labels, model, optimizer, args, iteration, device):

    inputs = inputs.to(device)
    labels = labels.to(device)

    # prediction
    y = model(inputs)

    loss = BCE_loss(y, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Iteration: {iteration}  Loss : {loss.data.cpu().numpy():.4f}")