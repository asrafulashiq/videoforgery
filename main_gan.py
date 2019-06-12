import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm

# custom module
import models
from models import Model, Model_boundary, Discriminator
import config
from dataset import Dataset_image
from utils import CustomTransform
from train import train, train_with_boundary, train_GAN
from test import test

import warnings
warnings.filterwarnings("ignore")

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

    # model name
    model_name = "GAN_" + args.model + "_" + args.model_type + "_" + \
                args.videoset + "_" + args.loss_type

    if args.boundary:
        model_name += "_boundary"

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
    fn_train = train_GAN

    model_G = Model(type=args.model_type)
    model_G = model_G.to(device)

    model_params = filter(lambda p: p.requires_grad, model_G.parameters())

    if args.model_type == "deeplab":
        model_params = [{
            'params': model_G.base.get_1x_lr_params(),
            'lr': args.lr / 10
        }, {
            'params': model_G.base.get_10x_lr_params(),
            'lr': args.lr
        }]

    # optimizer
    optimizer_G = torch.optim.Adam(model_params, lr=args.lr)


    model_D = Discriminator()
    model_D = model_D.to(device)

    optimizer_D = torch.optim.Adam(model_D.parameters(), lr=args.lr)

    model = [model_G, model_D]
    optimizer = [optimizer_G, optimizer_D]

    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    iteration = 1
    init_ep = 0

    # load coco pretrained on generator
    coco_pre_file = 'backup/base_unet_coco_bce.pkl'
    if os.path.exists(coco_pre_file):
        model_G.load_state_dict(
            torch.load(coco_pre_file)['model_state']
        )
        print("coco pretrained loaded in generator")

    # load if pretrained model
    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model_D.load_state_dict(checkpoint["model_state_D"])
        model_G.load_state_dict(checkpoint["model_state_G"])
        # optimizer.load_state_dict(checkpoint["opt_state"])
        # init_ep = checkpoint["epoch"]

    if args.validate:
        val_loader = dataset.load_data(args.batch_size*3, is_training=False,
                                       with_boundary=args.boundary)

    # train
    if args.test:  # test mode
        test(dataset, model_G, args, iteration, device, logger)
    else:  # train
        for ep in tqdm(range(init_ep, args.epoch)):
            # train
            for x_batch, y_batch, _ in dataset.load_data(
                    args.batch_size, is_training=True,
                    with_boundary=args.boundary):
                fn_train(x_batch, y_batch, model, optimizer, args, iteration,
                         device, logger)

                if args.validate and iteration % 10 == 0:
                    # validate
                    try:
                        x_val, y_val, _ = next(val_loader)
                    except StopIteration:
                        val_loader = dataset.load_data(args.batch_size, is_training=False,
                                                       with_boundary=args.boundary)
                        x_val, y_val, _ = next(val_loader)

                    with torch.no_grad():
                        fn_train(x_batch, y_batch, model, optimizer, args, iteration,
                                 device, logger, validate=True)

                iteration += 1
            # save current state
            torch.save(
                {
                    "epoch": ep,
                    "model_state_G": model_G.state_dict(),
                    "model_state_D": model_D.state_dict()                },
                "./ckpt/" + model_name + ".pkl",
            )

            # scheduler.step()

            # test
            test(dataset, model_G, args, iteration, device, logger, max_iter=500)

        logger.close()
