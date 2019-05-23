from torch import nn
from torch.nn import functional as F
import torch
import numpy as np


def bce_loss(y, labels):
    pass


def train(model, inputs, labels, args, iteration, device):
    y = model(inputs.to(device))

    print(iteration)
