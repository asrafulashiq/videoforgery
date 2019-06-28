"""main file for training bubblenet type comparison patch matching
"""

import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import skimage

# custom module
import config
from dataset import Dataset_image
from utils import CustomTransform, MultiPagePdf
from matching import tools

from train import train_template_match_im
from test import test_template_match_im


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_im_match()
    print(args)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #! TEMPORARY
    args.videoset = 'tmp_youtube'

    # model name
    model_name = args.model + "_" + args.model_type + "_" + \
        args.videoset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    logger = SummaryWriter("./logs/" + model_name)

    # dataset
    tsfm = CustomTransform

    dataset = Dataset_image(args=args, transform=tsfm)

    # model

    # model = tools.MatcherPair(patch_size=args.patch_size)
    model = tools.MatchUnet(im_size=args.size)
    model.to(device)

    iteration = 1
    init_ep = 0

    args.ckpt = "./ckpt/immatch_unet_tmp_youtubetmp.pkl"
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])

    model.to(device)

    root = Path("tmp_affinity")

    for ret in dataset.load_videos_all(is_training=False,
                                       shuffle=True, to_tensor=True):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        N = X.shape[0]
        if N > 15:
            ind = np.arange(0, N, 2)
        print(name, " : ", N)

        forge_time = np.arange(forge_time[0], forge_time[1]+1)
        gt_time = np.arange(gt_time[0], gt_time[1]+1)

        Data_corr = np.zeros((X.shape[0], X.shape[0], 400, 400))

        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        pdf = MultiPagePdf(total_im=N*N, out_name=str(path/"affinity.pdf"),
                           nrows=N, ncols=N, figsize=(16, 16))

        for i in range(N):
            for j in range(N):
                im1 = X[i]
                im2 = X[j]

                D_corr = model(im1.unsqueeze(
                    0).cuda(), im2.unsqueeze(0).cuda(), corr_only=True)
                D_corr = D_corr.squeeze()

                D_corr = D_corr.data.cpu().numpy()

                Data_corr[i, j] = D_corr

                pdf.plot_one(D_corr, cmap='Blues')

        pdf.final()
        break
