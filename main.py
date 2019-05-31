import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
# custom module

from models import Model
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train
from test import test


if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main()
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

    # freeze encoder block
    # for i, child in enumerate(model.unet.children()):
    #     if i < 11:
    #         for par in child.parameters():
    #             par.requires_grad = False

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
        test(dataset, model, args, iteration, device, logger)
    else:  # train
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for x_batch, y_batch, _ in dataset.load_data(args.batch_size, is_training=True):
                train(x_batch, y_batch, model, optimizer, args,
                    iteration, device, logger)
                iteration += 1
                # if iteration % 10 == 0:
                #     test(dataset, model, args, iteration, device, logger)

            # save current state
            torch.save({
                "epoch": ep,
                "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict()
            }, "./ckpt/"+args.model+"_"+args.videoset+".pkl")

            # test
            test(dataset, model, args, iteration, device, logger)

        logger.close()
