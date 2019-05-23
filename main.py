import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms

# custom module

from models import Model
from config import args
from dataset import Dataset_image
from utils import CustomTransform
from train import train

torch.set_default_tensor_type(torch.cuda.FloatTensor)


if __name__ == "__main__":

    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # dataset
    tsfm = CustomTransform()
    dataset = Dataset_image(args=args, transform=tsfm)

    # train test split
    train_len = int(0.8 * len(dataset))
    test_len = len(dataset) - train_len
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_len, test_len]
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False
    )

    # model
    model = Model()
    if torch.cuda.is_available():
        model.cuda()

    # load if pretrained model

    # train
    iteration = 0
    for x_batch, y_batch in train_loader:
        train(model, x_batch, y_batch, args, iteration)

    # validate

