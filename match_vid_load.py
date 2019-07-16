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


    root = Path("tmp_affinity")

    mask_processor = utils.Preprocessor(args)

    data_path = Path("./tmp_video_match")

    for fldr in data_path.iterdir():
        if not fldr.is_dir():
            continue
        path_data = fldr / "data_pred.pt"
        Data = torch.load(str(path_data))
        print("Data loaded from {}".format(path_data))

        X, Y_forge, Y_orig, gt_time, forge_time = Data['X'], \
            Data['Y_forge'], Data['Y_orig'], Data['gt_time'], \
            Data['forge_time']
        D_pred = Data['D_pred']
        name = Data['name']

        N = X.shape[0]
        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        pdf = MultiPagePdf(total_im=N*N*2, out_name=str(path / "affinity.pdf"),
                           nrows=N, ncols=2, figsize=(5, N*2))

        Hist = np.zeros((N, N))
        for i in range(N):
            if i in forge_time:
                i_ind = np.where(forge_time == i)[0][0]
                gt_ind = gt_time[i_ind]
            else:
                gt_ind = None
            out1 = D_pred[i, :, 0]
            out2 = D_pred[i, :, 1]
            out1 = out1.squeeze().data.cpu().numpy()
            out2 = out2.squeeze().data.cpu().numpy()

            for j in range(N):
                mask1 = out1[j]
                mask2 = out2[j]

                # preprocess mask
                mask1 = mask_processor.morph(mask1)
                mask2 = mask_processor.morph(mask2)

                if np.all(mask1 == 0):
                    mask2 = mask1
                if np.all(mask2 == 0):
                    mask1 = mask2

                im1 = tsfm.inverse(X[j])
                im2 = tsfm.inverse(X[i])
                im1_masked = im1 * mask1[..., None]
                im2_masked = im2 * mask2[..., None]

                # get histogram comparison
                vcomp = mask_processor.comp_hist(im1, im2, mask1, mask2,
                                                 compare_method=3)

                Hist[i, j] = 1 - vcomp

                ax = pdf.plot_one(im1_masked)
                ax.set_xlabel(f"{j}", fontsize="small")

                ax = pdf.plot_one(im2_masked)
                ax.set_xlabel(f"{i}", fontsize="small")

                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"Hist: {1-vcomp:.4f}")

                if gt_ind is not None and j == gt_ind:
                    ax.set_title("GT", fontsize="large")
        pdf.final()
        print("pdf saved")

