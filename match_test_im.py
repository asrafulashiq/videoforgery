
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

    args = config.arg_main_im_match()
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

    model = tools.MatchDeepLabV3p(im_size=args.size)
    model.to(device)

    iteration = 1
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])

    model.to(device)
    # model.eval()

    cnt = 1
    IOU_all = []

    cnt = 1
    for ret in dataset.load_data_template_match_pair(is_training=False, batch=True,
                                                     to_tensor=True):
        Xref, Xtem, Yref, Ytem, name = ret
        with torch.no_grad():
            model(Xref.to(device), Xtem.to(device))
        cnt += 1
        if cnt > 10:
            break

    cnt = 1
    model.eval() 

    for ret in dataset.load_data_template_match_pair(is_training=False, batch=True,
                                                     to_tensor=False):
        Xref, Xtem, Yref, Ytem, name = ret
        IOU = []
        print(f"{cnt} : {name}")
        cnt += 1
        print("-----------------")
        savepath = Path("tmp3") / name
        savepath.mkdir(parents=True, exist_ok=True)

        for k in range(Xref.shape[0]):
            # transform
            im_orig = Xref[k]
            im_forge = Xtem[k]
            im_ot = CustomTransform(size=args.size)(im_orig)
            im_ft = CustomTransform(size=args.size)(im_forge)

            im_ot = im_ot.unsqueeze(0).to(device)
            im_ft = im_ft.unsqueeze(0).to(device)
            with torch.no_grad():
                map_o, map_f = model(im_ot, im_ft)
            map_o = torch.sigmoid(map_o.squeeze())
            map_f = torch.sigmoid(map_f.squeeze())

            map_o = map_o.data.cpu().numpy()
            map_o = skimage.transform.resize(map_o, im_orig.shape[:2], order=0)

            map_f = map_f.data.cpu().numpy()
            map_f = skimage.transform.resize(map_f, im_orig.shape[:2], order=0)

            iou_s = tools.iou_mask_with_ignore(map_o,
                                               Yref[k], thres=args.thres)
            print(f"\t{k}: iou source: {iou_s}")

            iou_f = tools.iou_mask_with_ignore(map_f,
                                               Ytem[k], thres=args.thres)
            print(f"\t{k}: iou forge: {iou_f}\n")
            IOU.append([iou_s, iou_f])
            # draw

            fig, ax = plt.subplots(2, 3, figsize=(14, 8))

            imsrc = utils.add_overlay(im_orig, Yref[k] > 0.55, (Yref[k] < 0.55) & (Yref[k] > 0.45),
                                        c1=[0, 1, 0], c2=[0, 0.3, 0])

            imtem = utils.add_overlay(im_forge, Ytem[k] > 0.55, (Ytem[k] < 0.55) & (Ytem[k] > 0.45),
                                      c1=[0, 1, 0], c2=[0, 0.3, 0])
            ax[0, 0].imshow(imsrc)
            ax[1, 0].imshow(imtem)

            imsrc = utils.add_overlay(im_orig, map_o > args.thres,
                                      c1=[0, 0, 1])

            imtem = utils.add_overlay(im_forge, map_f > args.thres,
                                      c1=[0, 0, 1])
            ax[0, 1].imshow(imsrc)
            ax[1, 1].imshow(imtem)

            ax[0, 2].imshow(map_o, cmap='plasma')
            ax[1, 2].imshow(map_f, cmap='plasma')

            fname = savepath / f"{k}.png"
            fig.savefig(fname)
            plt.close('all')
        
        IOU = np.array(IOU)
        print("\n\tIou Mean source: ", np.mean(IOU[:, 0]))
        print("\n\tIou Mean fourge: ", np.mean(IOU[:, 1]))
        IOU_all.append([np.mean(IOU[:, 0]), np.mean(IOU[:, 1])])
    
    print("-------------")
    IOU_all = np.array(IOU_all)
    print("\n\tIou Mean source: ", np.mean(IOU_all[:, 0]))
    print("\n\tIou Mean fourge: ", np.mean(IOU_all[:, 1]))
