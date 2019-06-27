
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
from skimage import io

# custom module
import config
from dataset import Dataset_image
from utils import CustomTransform
from matching import tools
import utils

from train import train_template_match
from test import test_template_match


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_match()
    print(args)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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

    model = tools.MatcherPair()
    model.to(device)

    iteration = 1
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    model.eval()

    cnt = 1

    IOU = []
    for ret in dataset.load_videos_all(is_training=False, to_tensor=False):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        iou_all = []
        forge_time = np.arange(forge_time[0], forge_time[1] + 1)
        gt_time = np.arange(gt_time[0], gt_time[1] + 1)
        print(f"{cnt} : {name}")
        cnt += 1
        print("-----------------")
        savepath = Path("tmp3") / name
        savepath.mkdir(parents=True, exist_ok=True)
        for k in range(len(forge_time)):
            ind_forge = forge_time[k]
            ind_orig = gt_time[k]

            im_orig = X[ind_orig]
            im_forge = X[ind_forge]

            im_forge_mask = im_forge * Y_forge[ind_forge][..., None]
            im_orig_mask = im_orig * (1 - Y_forge[ind_orig])[..., None]

            # transform
            im_ot = CustomTransform(size=args.size)(im_orig_mask)
            im_ft = CustomTransform(size=args.patch_size)(im_forge_mask)

            im_ref = im_ot.unsqueeze(0).to(device)
            im_t = im_ft.unsqueeze(0).to(device)
            with torch.no_grad():
                map_o = torch.sigmoid(model(im_ref, im_t))
            map_o = map_o.squeeze()

            map_o = map_o.data.cpu().numpy()
            map_o = skimage.transform.resize(map_o, im_orig.shape[:2])

            iou = tools.iou_mask(map_o > args.thres, Y_orig[ind_orig] > 0.5)
            print(f"\t{k}: iou  {iou}")
            iou_all.append(iou)

            # draw
            imcopy = im_forge_mask
            imsrc = utils.add_overlay(im_orig, Y_orig[ind_orig],
                                      map_o>args.thres)

            fig, ax = plt.subplots(1, 3, figsize=(14, 8))
            ax[0].imshow(imcopy)
            ax[1].imshow(imsrc)
            ax[2].imshow(map_o, cmap='gray')
            fname = savepath / f"{k}.png"
            fig.savefig(fname)
            plt.close('all')
        print("\n\tIou Mean: ", np.mean(iou_all))
        IOU.append(np.mean(iou_all))
    print("-------------")
    print("Total: {:.4f}".format(np.mean(IOU)))
    print("-------------")
