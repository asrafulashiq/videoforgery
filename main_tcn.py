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
from tcn2d import TCN
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train, train_tcn
from test import test_track

import warnings
warnings.filterwarnings("ignore")

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

    # model name
    model_name = args.model + "_" + args.model_type + "_" + \
        args.videoset + "_" + args.loss_type

    print(f"Model Name: {model_name}")

    # logger
    logger = SummaryWriter("./logs/" + model_name)

    # dataset
    tsfm = CustomTransform(size=args.size)
    if args.videoset == 'coco':
        from dataset_coco import COCODataset
        dataset = COCODataset(args=args, transform=tsfm)
    else:
        dataset = Dataset_image(args=args, transform=tsfm)

    # model
    model = TCN(level=args.level)

    model = model.to(device)

    model_params = filter(lambda p: p.requires_grad, model.parameters())

    if args.model_type == "deeplab":
        model_params = [{
            'params': model.base.get_1x_lr_params(),
            'lr': args.lr / 10
        }, {
            'params': model.base.get_10x_lr_params(),
            'lr': args.lr
        }]

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    iteration = 1
    init_ep = 0
    # load if pretrained model
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["opt_state"])
        # init_ep = checkpoint["epoch"]

    if args.validate:
        val_loader = dataset.load_videos_track(
            is_training=False, add_prev=False)

    # train
    if args.test:  # test mode
        test_track(dataset, model, args, iteration,
                   device, logger=logger)
    else:  # train
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for ret in dataset.load_videos_track(
                    is_training=True, add_prev=False):
                X, Y_forge = ret
                train_tcn(X, Y_forge, model, optimizer, args, iteration,
                          device, logger)

                if args.validate and iteration % 10 == 0:
                    # validate
                    try:
                        ret = next(val_loader)
                    except StopIteration:
                        val_loader = dataset.load_videos_track(
                            is_training=False, add_prev=False)
                        ret = next(val_loader)

                    X, Y_forge = ret
                    with torch.no_grad():
                        train_tcn(X, Y_forge, model, optimizer, args, iteration,
                                  device, logger, validate=True)

                iteration += 1
            # save current state
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                },
                "./ckpt/" + model_name + ".pkl",
            )

            # scheduler.step()

            # test
            test_track(dataset, model, args, iteration,
                       device, logger=logger)

        logger.close()