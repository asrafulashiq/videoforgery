import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
# custom module

from models import Model_track as Model
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train
from test import test_track, test_track_video


if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_track()
    print(args)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logger
    logger = SummaryWriter("./logs/" + args.model+"_"+args.videoset)

    # dataset
    tsfm = CustomTransform(size=args.size)
    dataset = Dataset_image(args=args, transform=tsfm)

    # model
    model = Model().to(device)

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    iteration = 0
    init_ep = 0
    # load if pretrained model
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["opt_state"])
        init_ep = checkpoint["epoch"]

    # train
    if args.test:  # test mode
        test_track_video(dataset, model, args, iteration,
                         device, num=50, logger=logger)
    else:  # train
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for x_batch, y_batch in dataset.load_videos_track():
                train(x_batch, y_batch, model, optimizer, args,
                    iteration, device, logger)
                iteration += 1

                # if iteration % 1 == 0:
                #     test_track(dataset, model, args, iteration, device, 10, logger)

            # save current state
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict()
            }, "./ckpt/"+args.model+"_"+args.videoset+".pkl")

            # test
            test_track(dataset, model, args, iteration, device, num=20,
                        logger=logger)

        logger.close()
