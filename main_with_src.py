"""detection of both copy and move into single image file
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
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train, train_tcn
from test import test_tcn
from tcn2d import TCN, TCN2

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

    args.loss_type = ""
    # model name
    model_name = args.model + "_" + args.model_type + "_" + \
        args.videoset + args.suffix

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

    model = TCN2(level=args.level)

    iteration = 1
    init_ep = 0
    # load if pretrained model

    # load coco pretrained
    coco_path = './ckpt/tcn_coco_unet_coco_bce.pkl'
    if os.path.exists(coco_path):
        coco_state = torch.load(coco_path)
        model.tcn_forge.load_state_dict(coco_state['model_state'])
        print("COCO pretrained loaded on forge part")


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

    model_params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer
    optimizer = torch.optim.Adam(model_params, lr=args.lr)

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)


    if args.validate:
        val_loader = dataset.load_videos_all(is_training=False)

    # train
    if args.test:  # test mode
        test_tcn(dataset, model, args, iteration, device,
                 logger=logger, num=None,  with_src=True)
    else:  # train
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for ret in dataset.load_videos_all(is_training=True):
                X, Y_forge, forge_time, Y_orig, gt_time, name = ret
                x_batch = X
                y_batch = torch.cat((Y_forge, Y_orig), 1)
                train_tcn(x_batch, y_batch, model, optimizer, args, iteration,
                          device, logger, with_src=True)

                if args.validate and iteration % 10 == 0:
                    # validate
                    try:
                        ret = next(val_loader)
                    except StopIteration:
                        val_loader = dataset.load_videos_all(is_training=False)
                        ret = next(val_loader)
                    X, Y_forge, forge_time, Y_orig, gt_time, name = ret
                    x_batch = X
                    y_batch = torch.cat((Y_forge, Y_orig), 1)

                    with torch.no_grad():
                        train_tcn(x_batch, y_batch, model, optimizer, args, iteration,
                                  device, logger, with_src=True, validate=True)

                iteration += 1
            # save current state
            torch.save(
                {
                    "epoch": ep,
                    "model_state": model.module.state_dict()
                        if isinstance(model, nn.DataParallel)
                        else model.state_dict(),
                    "opt_state": optimizer.state_dict(),
                },
                "./ckpt/" + model_name + ".pkl",
            )

            # scheduler.step()

            # test
            if (ep + 1) % 5 == 0:
                num = None
                print("TEST ALL DATA !!!!!!!!!!")
            else:
                num = 10
            test_tcn(dataset, model, args, iteration, device,
                     logger=logger, num=num, with_src=True)
        logger.close()
