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
from tqdm import tqdm

# metric
from sklearn import metrics

import utils
from utils import MultiPagePdf

def test(dataset, model, args, iteration, device, logger=None):
    model.eval()

    aucs = []
    f1s = []

    # counter = 0
    for X, labels, info in tqdm(dataset.load_data(batch=40, is_training=False)):
        with torch.no_grad():
            X = X.to(device)
            labels = labels.to(device)
            preds = model(X)
            preds = torch.sigmoid(preds)

            preds = preds.squeeze().data.cpu().numpy()
            labels = labels.squeeze().data.cpu().numpy()
            labels = (labels > 0).astype(np.float32)
        _auc, _f1 = score_report(preds.flatten(), labels.flatten(), args, iteration)
        aucs.append(_auc)
        f1s.append(_f1)

        # plot_samples(preds.data.cpu().numpy(), labels.data.cpu().numpy(), args, info)
        # break
    # score_report(Y_pred, Y_gt, args, iteration, logger)

    auc_mean = np.mean(aucs)
    f1_mean = np.mean(f1s)

    print("TEST")
    print(f"AUC_ROC: {auc_mean: .4f}")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)
        logger.add_scalar("score/auc_roc", auc_mean, iteration)


def score_report(y_pred, y_gt, args, iteration, logger=None):
    # ROC AUC
    auc_roc = metrics.roc_auc_score(y_gt, y_pred)

    # f1 score
    f1_score = metrics.f1_score(y_gt, [pr > args.thres for pr in y_pred])

    return auc_roc, f1_score



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



