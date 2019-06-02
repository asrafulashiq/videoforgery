"""main file for training bubblenet type comparison patch matching
"""

import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

# custom module
from models import Model_comp
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train_match_in_the_video
from test import test_match_in_the_video


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    args = config.arg_main_search()

    tsfm = CustomTransform(size=args.size)
    dataset = Dataset_image(args=args, transform=tsfm)

    for i in range(10):
        test_match_in_the_video(dataset, args, i)
