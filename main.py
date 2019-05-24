import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter

# custom module

from models import Model
from config import args
from dataset import Dataset_image
from utils import CustomTransform
from train import train
from test import test

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # logger
    logger = SummaryWriter("./logs/" + args.model)

    # dataset
    tsfm = CustomTransform()
    dataset = Dataset_image(args=args, transform=tsfm)

    # train test split
    train_len = int(args.split * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=60, shuffle=True
    )

    # model
    model = Model().to(device)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    iteration = 0
    init_ep = 0


    # load if pretrained model
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["opt_state"])
        init_ep = checkpoint["epoch"]

    # train
    for ep in range(init_ep, args.epoch):
        # train
        for x_batch, y_batch, _ in train_loader:
            train(x_batch, y_batch, model, optimizer, args,
                iteration, device, logger)
            iteration += 1
            # break

        # save current state
        torch.save({
            "epoch": ep,
            "model_state": model.state_dict(),
            "opt_state": optimizer.state_dict()
        }, "./ckpt/"+args.model+".pkl")

        # test
        test(test_loader, model, args, iteration, device, logger)

    logger.close()

        # validate

