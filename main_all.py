import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from skimage import io
import cv2
from pathlib import Path
import skimage
# custom module
from models import Model
import config
from dataset import Dataset_image
from utils import CustomTransform
import utils
from train import train
from test import test


if __name__ == "__main__":

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main()
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # dataset
    tsfm = CustomTransform(size=args.size)
    dataset = Dataset_image(args=args, transform=tsfm)

    model = Model().to(device)

    if args.ckpt is not None:
        checkpoint = torch.load(args.ckpt)
        model.load_state_dict(checkpoint["model_state"])

    for cnt, dat in enumerate(dataset.load_videos_all()):
        # work with a particular video
        X, Y_red, forge_time, Y_green, gt_time, vid_name = dat

        Pred = np.zeros(X.shape[:3])

        for i in tqdm(range(X.shape[0])):
            x_im = X[i]
            x_im = tsfm(x_im.astype(np.float32)).to(device)
            pred = model(x_im.unsqueeze(0))
            pred = torch.sigmoid(pred)

            pred = pred.squeeze().data.cpu().numpy()
            pred = (pred > args.thres).astype(np.float)

            kernel = np.ones((5,5))
            pred = cv2.morphologyEx(pred, cv2.MORPH_OPEN, kernel)
            pred = cv2.morphologyEx(pred, cv2.MORPH_CLOSE, kernel)

            # get the largest connected component
            pred_lab = skimage.measure.label(pred, background=0)

            labels = np.unique(pred_lab)
            if labels.size > 1:
                area = [(i, np.sum(pred_lab==i)) for i in labels if i != 0]
                max_lab, _ = max(area, key = lambda x: x[1])
                pred_lab[pred_lab != max_lab] = 0
                pred_lab[pred_lab == max_lab] = 1
                pred_lab = pred_lab.astype(np.float)

                # if np.sum(pred_lab) / (args.size**2) < 0.05:
                #     pred_lab = np.zeros_like(pred_lab)

            Pred[i] = pred_lab

            #! TEST
            pp = utils.image_with_mask(X[i], pred_lab, type="foreground")
            path = Path(f"tmp/{cnt}")
            path.mkdir(parents=True, exist_ok=True)
            io.imsave(str(path/f"{i}.jpg"), pp)


