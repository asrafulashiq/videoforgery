from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
from tqdm import tqdm
import skimage
import cv2
import shutil

from config import args
from dataset import Dataset_image
from utils import CustomTransform, add_overlay
from models import Model


if __name__ == "__main__":
    if args.ckpt is None:
        raise SystemExit

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    np.random.seed(args.seed)

    model = Model()
    model.to(device)

    checkpoint = torch.load(args.ckpt)
    model.load_state_dict(checkpoint["model_state"])

    dataset = Dataset_image(args=args, transform=None)
    tsfm = CustomTransform()

    if not os.path.exists("./tmp"):
        os.mkdir("./tmp")

    for k in range(20):
        path = "./tmp/{}".format(k)
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        for i, (image, label) in tqdm(enumerate(dataset.get_frames_from_video())):
            im_tensor = tsfm(image)
            im_tensor = im_tensor.to(device)

            with torch.no_grad():
                im_tensor = im_tensor.unsqueeze(0)
                output = model(im_tensor)
                output = output.squeeze()

            output = output.data.cpu().numpy()
            output = (output > args.thres).astype(np.float32)

            pred = skimage.transform.resize(output, image.shape[:2])

            # overlay two mask
            nim = add_overlay(image, label, pred)
            nim = skimage.img_as_ubyte(nim)

            skimage.io.imsave(
                os.path.join(path, f"{i}.jpg"),
                nim
            )
