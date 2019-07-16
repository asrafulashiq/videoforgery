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
from matching import tools, tools_argmax

from train import train_template_match_im
from test import test_template_match_im

import utils


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
    tsfm = CustomTransform(size=args.size)

    dataset = Dataset_image(args=args, transform=tsfm)

    # model
    model = tools_argmax.BusterModel()
    model.to(device)

    iteration = 1
    init_ep = 0

    args.ckpt = "ckpt/immatch_deeplab_tmp_youtube_stf_mix.pkl"
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])

    model.to(device)

    root = Path("tmp_video_match")

    mask_processor = utils.Preprocessor(args)

    for ret in tqdm(dataset.load_videos_all(is_training=False,
                                            shuffle=True, to_tensor=True)):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        N = X.shape[0]
        if N > 20:
            continue
        print(name, " : ", N)

        forge_time = np.arange(forge_time[0], forge_time[1]+1)
        gt_time = np.arange(gt_time[0], gt_time[1]+1)

        Data_corr = np.zeros((X.shape[0], X.shape[0], 40**2, 40**2))

        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        D_pred = torch.zeros((N, N, 2, args.size, args.size))
        for i in range(N):
            Xr = X.to(device)
            Xt = X[[i]].repeat((Xr.shape[0], 1, 1, 1)).to(device)

            if i in forge_time:
                i_ind = np.where(forge_time == i)[0][0]
                gt_ind = gt_time[i_ind]
            else:
                gt_ind = None

            with torch.no_grad():
                out1, out2 = model(Xr, Xt)
            D_pred[i, :, 0] = out1.squeeze()
            D_pred[i, :, 1] = out2.squeeze()

        torch.save(
            {
                "X": X,
                "Y_forge": Y_forge,
                "Y_orig": Y_orig,
                "forge_time": forge_time,
                "gt_time": gt_time,
                "D_pred": D_pred, 
                "name": name
            },
            str(path / "data_pred.pt")
        )
