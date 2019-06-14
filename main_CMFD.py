"""main file for image forgery detection
"""
import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

# custom module
import models
from models import Model, Model_boundary
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train, train_with_boundary
from test import test, test_CMFD

import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main()

    root = '/home/islama6a/local/CMFD_PM_video/tmp2_utube'

    print(args)
    dataset = Dataset_image(args=args, transform=None)

    test_CMFD(dataset, args, root)
