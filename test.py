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

import warnings

warnings.filterwarnings("ignore")


@torch.no_grad()
def test_track(dataset, model, args, iteration, device, num=None, logger=None):
    model.eval()

    counter = 0
    Tp, Tn, Fp, Fn = 0, 0, 0, 0
    for cnt, (X_all, Y_all) in tqdm(
            enumerate(dataset.load_videos_track(is_training=False))):
        # counter = 0
        prev = None
        _len = X_all.shape[0]
        for i in range(_len):
            X = X_all[i, :3]
            labels = Y_all[i]

            X = X.to(device)
            labels = labels.to(device)
            if i == 0:
                prev = torch.zeros(labels.shape,
                                   dtype=torch.float32).to(device)

            inp = torch.cat((X, prev), 0)
            preds = model(inp.unsqueeze(0))
            preds = torch.sigmoid(preds)
            prev = preds.squeeze(0)

            _preds = preds.squeeze().data.cpu().numpy().flatten()
            _labels = labels.squeeze().data.cpu().numpy().flatten()
            _labels = (_labels > 0.5).astype(np.float32)
            P.extend(_preds.tolist())
            L.extend(_labels.tolist())

            tn, fp, fn, tp = metrics.confusion_matrix(
                _labels, (_preds > args.thres)).ravel()
            Tp, Tn, Fp, Fn = Tp + tp, Tn + tn, Fp + fp, Fn + fn

        if num is not None and cnt >= num:
            break

    f1_mean = 2 * Tp / (2 * Tp + Fp + Fn)
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)


@torch.no_grad()
def test_CMFD(dataset, args, root, num=None, logger=None):
    from scipy.io import loadmat

    Tscore = np.zeros(4)

    for cnt, ret in enumerate(tqdm(dataset.load_videos_all())):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        matfile = os.path.join(
            root, name+".mat"
        )
        assert os.path.exists(matfile)
        predY = loadmat(matfile)['map']
        predY = predY.transpose((2, 0, 1))

        for i in range(X.shape[0]):
            pred_o = predY[i]
            labels  = (Y_forge[i] > 0) & (Y_orig[i] > 0)
            pred = skimage.transform.resize(pred_o, labels.shape[:2])

            tt = metrics.confusion_matrix(
                labels.ravel(), (pred > args.thres).ravel()).ravel()
            Tscore += np.array(tt)

    f1_mean = utils.fscore(Tscore)
    print()
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")

@torch.no_grad()
def test_move(dataset, model, args, iteration, device, num=None, logger=None):
    model.eval()

    aucs = []
    f1s = []
    P = []
    L = []

    counter = 0

    for cnt, (X_all, Y_all) in tqdm(
            enumerate(dataset.load_videos_track(is_training=False))):
        # counter = 0
        prev = None
        _len = X_all.shape[0]
        for i in range(_len):
            X = X_all[i, :3]
            labels = Y_all[i]

            X = X.to(device)
            labels = labels.to(device)
            if i == 0:
                prev = torch.zeros(labels.shape,
                                   dtype=torch.float32).to(device)

            inp = torch.cat((X, prev), 0)
            preds = model(inp.unsqueeze(0))
            preds = torch.sigmoid(preds)
            prev = preds.squeeze(0)

            _preds = preds.squeeze().data.cpu().numpy().flatten()
            _labels = labels.squeeze().data.cpu().numpy().flatten()
            _labels = (_labels > 0.5).astype(np.float32)
            P.extend(_preds.tolist())
            L.extend(_labels.tolist())

        if num is not None and cnt >= num:
            break

    if len(P) > 5 * 4e5:
        _auc, _f1 = score_report(P, L, args, iteration)
        aucs.append(_auc)
        f1s.append(_f1)

    auc_mean = np.mean(aucs)
    f1_mean = np.mean(f1s)

    print("TEST")
    print(f"AUC_ROC: {auc_mean: .4f}")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)
        logger.add_scalar("score/auc_roc", auc_mean, iteration)


@torch.no_grad()
def test_track_video(dataset,
                     model,
                     args,
                     iteration,
                     device,
                     num=10,
                     logger=None):

    for k in range(25):
        path = "./tmp/{}".format(k)

        tsfm = CustomTransform(args.size)
        for i, (image,
                label) in tqdm(enumerate(dataset.get_frames_from_video())):
            im_tensor = tsfm(image)
            im_tensor = im_tensor.to(device)

            if i == 0:
                prev = torch.zeros((1, args.size, args.size),
                                   dtype=im_tensor.dtype).to(device)
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


def test_match_in_the_video(dataset, args, tk=3):
    """match an image with forged match in the video."""

    matcher = utils.TemplateMatch(thres=args.thres)

    _top = []

    for cnt, tmp in tqdm(enumerate(dataset.get_search_from_video())):
        X, x_ref, gt_ind, first_ind = tmp

        sim_list = []
        for i in range(X.shape[0]):
            out = matcher.match(X[i], x_ref)
            bbox, val, patched_im = out
            sim_list.append(val)

            if i == gt_ind:
                import pdb

            imname = "tmp2/{}/{}.jpg".format(cnt, i)
            Path(imname).parent.mkdir(parents=True, exist_ok=True)
            # skimage.io.imsave(imname, patched_im)

        sort_ind = np.argsort(sim_list)[::-1]
        k = np.where(sort_ind == gt_ind)[0][0] + 1
        print("{:03d} Target: {:2d}, Matched: {:2d}, topk: {:3d}, GT: {:2d}".
              format(cnt, first_ind, sort_ind[0], k, gt_ind))

        _top.append(k)

    # top-k score
    print("-----------------------")

    for fl in range(1, 6):
        sc = [1 if i <= fl else 0 for i in _top]
        print("Top-{} accuracy: {:.2f}".format(fl, np.sum(sc) / len(sc)))


@torch.no_grad()
def test(dataset, model, args, iteration, device, logger=None, max_iter=None):
    model.eval()
    counter = 0

    Tscore = np.zeros(4)

    for X, labels, info in tqdm(dataset.load_data(batch=40,
                                                  is_training=False)):
        X = X.to(device)
        labels = labels.to(device)
        preds = model(X)
        if args.boundary:
            preds = preds[0]
        preds = torch.sigmoid(preds)

        preds = preds.squeeze().data.cpu().numpy()
        labels = labels.squeeze().data.cpu().numpy()
        labels = (labels > 0.5).astype(np.float32)

        tt = metrics.confusion_matrix(
            labels.ravel(), (preds > args.thres).ravel()).ravel()
        Tscore += np.array(tt)
        counter += X.shape[0]

        if max_iter is not None and counter > max_iter:
            break

    f1_mean = utils.fscore(Tscore)
    print()
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)


@torch.no_grad()
def test_with_src(dataset,
                  model,
                  args,
                  iteration,
                  device,
                  logger=None,
                  max_iter=None):
    model.eval()
    counter = 0

    Tforge = np.array([0, 0, 0, 0])
    Tsrc = np.array([0, 0, 0, 0])
    Tback = np.array([0, 0, 0, 0])

    for X, labels, info in tqdm(
            dataset.load_data_with_src(batch=40, is_training=False)):
        X = X.to(device)
        labels = labels.to(device)
        preds = model(X)
        if args.boundary:
            preds = preds[0]
        preds = torch.sigmoid(preds)

        preds = preds.data.cpu().numpy()
        labels = labels.data.cpu().numpy()
        # labels = (labels > 0.5).astype(np.float32)

        preds_bool = preds > args.thres

        tforg = metrics.confusion_matrix((labels == 2).ravel(),
                                         preds_bool[:, 2].ravel()).ravel()

        tsrc = metrics.confusion_matrix((labels == 1).ravel(),
                                        preds_bool[:, 1].ravel()).ravel()

        tback = metrics.confusion_matrix((labels == 0).ravel(),
                                         preds_bool[:, 0].ravel()).ravel()

        Tforge += tforg
        Tsrc += tsrc
        Tback += tback

        counter += X.shape[0]

        if max_iter is not None and counter > max_iter:
            break

    f1_src = utils.fscore(Tsrc)
    recall_src = Tsrc[3] / (Tsrc[3] + Tsrc[2])

    f1_forge = utils.fscore(Tforge)
    f1_back = utils.fscore(Tback)
    print()
    print("TEST")
    print(
        f"F1 score - forge: {f1_forge:.4f}, src: {f1_src:.4f}, back: {f1_back:.4f}"
        + f" Recall: src: {recall_src:.4f}"
    )

    if logger is not None:
        logger.add_scalar("score/f1_src", f1_src, iteration)
        logger.add_scalar("score/f1_forge", f1_forge, iteration)
        logger.add_scalar("score/f1_back", f1_back, iteration)


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
        image = skimage.transform.resize(image,
                                         (label.shape[0], label.shape[1]))
        image = skimage.img_as_float32(image)

        # im = utils.overlay_masks(label, pred > args.thres)
        im = utils.add_overlay(image, label, pred > args.thres)
        pdf.plot_one(im)

    pdf.final()
    print(f"{out_file_name} created")
