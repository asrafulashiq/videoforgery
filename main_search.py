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

from vid_match import match_in_the_video

if __name__ == "__main__":

    args = config.arg_main_search()

    dataset = Dataset_image(args=args, transform=None)

    match_in_the_video(dataset, args)
