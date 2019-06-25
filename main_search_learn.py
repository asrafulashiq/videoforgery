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
from utils import CustomTransform
from matching import tools


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_tcn()
    print(args)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    args.loss_type = ""
    # model name
    args.model = "match"
    model_name = args.model + "_" + args.model_type + "_" + \
        args.videoset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    logger = SummaryWriter("./logs/" + model_name)

    # dataset
    tsfm = CustomTransform

    dataset = Dataset_image(args=args, transform=tsfm)

    # model

    model = tools.MatcherPair(type='resnet')
    model.to(device)

    iteration = 1
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        # if isinstance(model, nn.DataParallel):
        #     model.module.load_state_dict(checkpoint["model_state"])
        # else:
        model.load_state_dict(checkpoint["model_state"])

    if torch.cuda.device_count() > 1:
        print("Using ", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # model_params = filter(lambda p: p.requires_grad, model.parameters())
    # # optimizer
    # optimizer = torch.optim.Adam(model_params, lr=args.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    model.eval()
    for ep in tqdm(range(init_ep, args.epoch)):
        # train
        for ret in dataset.load_videos_all(is_training=False, to_tensor=False):
            X, Y_forge, forge_time, Y_orig, gt_time, name = ret
            
            iou_all = []
            forge_time = np.arange(forge_time[0], forge_time[1] + 1)
            gt_time = np.arange(gt_time[0], gt_time[1] + 1)
            print(f"{name}")
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

                bb_forge = tools.get_bbox(Y_forge[ind_forge] > 0.5)
                bb_orig = tools.get_bbox(Y_orig[ind_orig] > 0.5)

                x, y, w, h = bb_forge
                im_f = im_forge_mask[y:y+h, x:x+w]

                # transform
                im_ot = tsfm(size=(240, 240))(im_orig_mask)
                im_ft = tsfm(size=(48, 48))(im_f)

                im_ref = im_ot.unsqueeze(0).to(device)
                im_t = im_ft.unsqueeze(0).to(device)
                with torch.no_grad():
                    map_o = model(im_ref, im_t)
                map_o = map_o.squeeze()

                map_o = map_o.data.cpu().numpy()
                map_o = skimage.transform.resize(map_o, im_orig.shape[:2])
                map_o[bb_forge[1]:bb_forge[1]+bb_forge[3],
                      bb_forge[0]:bb_forge[0]+bb_forge[2]] = 0
                pred_bbox = tools.locate_bbox(map_o, w, h)

                iou = tools.IoU(pred_bbox, bb_orig)
                print(f"\t{k}: iou  {iou}")
                iou_all.append(iou)

                # draw
                imcopy = tools.draw_rect(im_forge_mask, bb_forge)
                imsrc = tools.draw_rect(im_orig_mask, pred_bbox, bb_orig
                
                )

                fig, ax = plt.subplots(1, 3, figsize=(14, 8))
                ax[0].imshow(imcopy)
                ax[1].imshow(imsrc)
                ax[2].imshow(map_o, cmap='gray')
                fname = savepath / f"{k}.png"
                fig.savefig(fname)
            plt.close('all')

            print("\n\t Iou Mean: ", np.mean(iou_all))
