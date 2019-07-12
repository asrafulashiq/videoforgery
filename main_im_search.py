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
from dataset_coco import COCODataset
from utils import CustomTransform
from matching import tools, tools_argmax

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
    # args.videoset = 'tmp_youtube'

    # model name
    model_name = args.model + "_" + args.model_type + "_" + \
        args.videoset + args.suffix

    print(f"Model Name: {model_name}")

    # logger
    logger = SummaryWriter("./logs/" + model_name)

    # dataset
    tsfm = CustomTransform(size=args.size)

    if args.videoset == 'coco':
        dataset = COCODataset(args=args, transform=tsfm)
    else:
        dataset = Dataset_image(args=args, transform=tsfm)

    # model

    # model = tools.MatcherPair(patch_size=args.patch_size)
    # model = tools.MatchUnet(im_size=args.size)
    model = tools_argmax.BusterModel()
    model.to(device)

    iteration = args.resume
    init_ep = 0

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"], strict=False)

    model_params = [
        {"params": model.get_1x_lr_params(), "lr": args.lr / 10},
        {"params": model.get_10x_lr_params(), "lr": args.lr * 10},
    ]

    # model_params = filter(lambda p: p.requires_grad, model.parameters())
    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    model.to(device)

    if args.test:
        test_template_match_im(dataset, model, args, iteration, device,
                               logger=logger, num=None)
    else:
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for ret in dataset.load_data_template_match_pair(is_training=True, batch=True,
                                                             to_tensor=True):
                Xref, Xtem, Yref, Ytem, name = ret
                train_template_match_im(Xref, Xtem, Yref, Ytem, model, optimizer, args,
                                        iteration, device, logger=logger)
                iteration += 1

            test_template_match_im(dataset, model, args, iteration, device,
                                   logger=logger, num=10)
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) \
                    else model.state_dict()
            torch.save(
                {
                    "epoch": ep,
                    "model_state": state,
                    "opt_state": optimizer.state_dict(),
                },
                "./ckpt/" + model_name + ".pkl",
            )
            scheduler.step()

    logger.close()
