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
from utils import CustomTransform
from train import train_match_in_the_video
from test import test_match_in_the_video


if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_search()

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logger
    logger = SummaryWriter("./logs/" + args.model + "_" + args.videoset)

    # dataset
    tsfm = CustomTransform(size=args.size)
    dataset = Dataset_image(args=args, transform=tsfm)

    # model
    model = Model_comp().to(device)

    # freeze resnet module
    # for param in model.res_extractor.parameters():
        # param.requires_grad = False

    # optimizer
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,
        momentum=0.95
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[2000, 5000, 10000], gamma=0.5
    )

    iteration = 0
    init_ep = 0
    # load if pretrained model
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["opt_state"])
        init_ep = checkpoint["epoch"]

        scheduler.last_epoch = init_ep

    # train
    if args.test:  # test mode
        test_match_in_the_video(dataset, model, args, iteration, device, logger)
    else:  # train
        for itr in tqdm(range(init_ep, args.epoch)):
            # train
            train_match_in_the_video(
                dataset, model, optimizer, args, itr, device, logger
            )

            scheduler.step()

            if (itr + 1) % 1000 == 0:  # save model
                # save current state
                torch.save(
                    {
                        "epoch": itr,
                        "model_state": model.state_dict(),
                        "opt_state": optimizer.state_dict(),
                    },
                    "./ckpt/" + args.model + "_" + args.videoset + ".pkl",
                )

            if itr % 1000 == 0:
                # test
                test_match_in_the_video(dataset, model, args, iteration, device, logger)

        logger.close()
