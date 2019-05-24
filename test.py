from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import os

import utils
from utils import MultiPagePdf

def test(test_loader, model, args, iteration, device, logger=None):
    model.eval()
    (X, labels, info) = next(iter(test_loader))
    with torch.no_grad():
        X = X.to(device)
        labels = labels.to(device)
        preds = model(X)
        preds = torch.sigmoid(preds)

        preds = preds.squeeze()
        labels = labels.squeeze()

    plot_samples(preds.data.cpu().numpy(), labels.data.cpu().numpy(), args, info)


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
        inf = (info[0][i], info[1][i])
        im = utils.overlay_masks(label, pred > args.thres)
        pdf.plot_one(im)

    pdf.final()
    print(f"{out_file_name} created")



