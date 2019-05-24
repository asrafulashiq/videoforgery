from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage
import os

import utils
from utils import MultiPagePdf

def test(dataset, model, args, iteration, device, logger=None):
    model.eval()
    for X, labels, info in dataset.load_data(batch=40, is_training=False):
        with torch.no_grad():
            X = X.to(device)
            labels = labels.to(device)
            preds = model(X)
            preds = torch.sigmoid(preds)

            preds = preds.squeeze()
            labels = labels.squeeze()

        plot_samples(preds.data.cpu().numpy(), labels.data.cpu().numpy(), args, info)
        break


def plot_samples(preds, labels, args, info=None):
    num = preds.shape[0]
    out_file_name = args.test_path + "/sample.pdf"
    if not os.path.exists(args.test_path):
        os.mkdir(args.test_path)

    pdf = MultiPagePdf(num, out_file_name)

    for i in range(num):
        pred = preds[i]
        # if np.any(pred>0.5):
        #     import pdb
        #     pdb.set_trace()
        label = labels[i]
        imfile = info[i][0]
        image = skimage.io.imread(imfile)
        image = skimage.transform.resize(image, (label.shape[0], label.shape[1]))
        image = skimage.img_as_float32(image)

        # im = utils.overlay_masks(label, pred > args.thres)
        im = utils.add_overlay(image, label, pred>args.thres)
        pdf.plot_one(im)

    pdf.final()
    print(f"{out_file_name} created")



