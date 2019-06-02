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
from utils import MultiPagePdf, add_overlay
from utils import CustomTransform


@torch.no_grad()
def test_track(dataset, model, args, iteration, device, num=10, logger=None):
    model.eval()

    aucs = []
    f1s = []
    P = []
    L = []
    for i in tqdm(range(num)):
        # counter = 0
        prev = None
        for X, labels in dataset.get_frames_from_video(do_transform=True):
            X = X.to(device)
            labels = labels.to(device)
            if prev is None:
                prev = torch.zeros(labels.shape, dtype=torch.float32).to(device)

            inp = torch.cat((X, prev), 0)
            preds = model(inp.unsqueeze(0))
            preds = torch.sigmoid(preds)
            prev = preds.squeeze(0)

            _preds = preds.squeeze().data.cpu().numpy().flatten()
            _labels = labels.squeeze().data.cpu().numpy().flatten()
            _labels = (_labels > 0.5).astype(np.float32)
            P.extend(_preds.tolist())
            L.extend(_labels.tolist())

        # _auc, _f1 = score_report(P, L, args, iteration)
        # aucs.append(_auc)
        # f1s.append(_f1)
    auc_mean, f1_mean = score_report(P, L, args, iteration)

    # auc_mean = np.mean(aucs)
    # f1_mean = np.mean(f1s)

    print("TEST")
    print(f"AUC_ROC: {auc_mean: .4f}")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)
        logger.add_scalar("score/auc_roc", auc_mean, iteration)


@torch.no_grad()
def test_track_video(dataset, model, args, iteration, device, num=10, logger=None):

    for k in range(25):
        path = "./tmp/{}".format(k)

        tsfm = CustomTransform(args.size)
        for i, (image, label) in tqdm(enumerate(dataset.get_frames_from_video())):
            im_tensor = tsfm(image)
            im_tensor = im_tensor.to(device)

            if i == 0:
                prev = torch.zeros((1, args.size, args.size), dtype=im_tensor.dtype).to(
                    device
                )
            X = torch.cat((im_tensor, prev), 0)

            output = model(X.unsqueeze(0))
            output = torch.sigmoid(output)
            prev = output.reshape((1, args.size, args.size)).clone()

            output = output.squeeze()

            output = output.data.cpu().numpy()
            output = (output > args.thres).astype(np.float32)

            pred = skimage.transform.resize(output, image.shape[:2])

            # overlay two mask
            nim = add_overlay(image, label, pred)
            nim = skimage.img_as_ubyte(nim)

            Path(path).mkdir(exist_ok=True, parents=True)
            skimage.io.imsave(os.path.join(path, f"{i}.jpg"), nim)


@torch.no_grad()
def test_match_in_the_video(dataset, args, iteration):
    """match an image with forged match in the video
    """

    matcher = utils.TemplateMatch(thres=args.thres)

    for cnt in range(10):
        X, x_ref, gt_ind, first_ind = dataset.get_search_from_video()

        # X = X.to(device)
        # x_ref = x_ref.to(device)

        sim_list = []
        for i in range(X.shape[0]):
            # _input = torch.stack([x_ref, X[i]], 0).to(device).unsqueeze(0)
            # out = model(_input).squeeze()
            # out = torch.sigmoid(out)
            out = matcher.match(X[i], x_ref)
            bbox, val, patched_im = out
            sim_list.append(val)

        sort_ind = np.argsort(sim_list)
        print(
            "{:03d} Target: {:2d}, Matched: {:2d}, topk: {:3d}, GT: {:2d}".format(
                cnt,
                first_ind,
                sort_ind[0],
                np.where(sort_ind == gt_ind)[0][0] + 1,
                gt_ind,
            )
        )


@torch.no_grad()
def test(dataset, model, args, iteration, device, logger=None):
    model.eval()

    aucs = []
    f1s = []

    counter = 0
    P = []
    L = []
    for X, labels, info in tqdm(dataset.load_data(batch=40, is_training=False)):
        X = X.to(device)
        labels = labels.to(device)
        preds = model(X)
        preds = torch.sigmoid(preds)

        preds = preds.squeeze().data.cpu().numpy()
        labels = labels.squeeze().data.cpu().numpy()
        labels = (labels > 0).astype(np.float32)

        P.extend(preds.flatten().tolist())
        L.extend(labels.flatten().tolist())

        # _auc, _f1 = score_report(preds.flatten(), labels.flatten(), args, iteration)
        # aucs.append(_auc)
        # f1s.append(_f1)

        counter += 1
        if counter % 15 == 0:
            _auc, _f1 = score_report(P, L, args, iteration)
            aucs.append(_auc)
            f1s.append(_f1)
            P = []
            L = []

        # plot_samples(preds.data.cpu().numpy(), labels.data.cpu().numpy(), args, info)
        # break
    # score_report(Y_pred, Y_gt, args, iteration, logger)
    if len(P) > 4e5:
        _auc, _f1 = score_report(P, L, args, iteration)
        aucs.append(_auc)
        f1s.append(_f1)

    auc_mean = np.mean(aucs)
    f1_mean = np.mean(f1s)

    print()
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
        im = utils.add_overlay(image, label, pred > args.thres)
        pdf.plot_one(im)

    pdf.final()
    print(f"{out_file_name} created")

