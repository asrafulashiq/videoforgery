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
from matching import tools
import warnings

warnings.filterwarnings("ignore")


@torch.no_grad()
def test_track(dataset, model, args, iteration, device, logger=None, num=None):
    model.eval()

    Tscore = np.zeros(4)
    for cnt, (X_all, Y_all) in tqdm(
            enumerate(dataset.load_videos_track(is_training=False, add_prev=False))):

        inputs = X_all.to(device)
        labels = Y_all

        preds = model(inputs)
        preds = torch.sigmoid(preds)

        f_preds = preds.squeeze().data.cpu().numpy().flatten()
        f_labels = labels.squeeze().data.cpu().numpy().flatten()
        f_labels = (f_labels > 0.5).astype(np.float32)

        tt = utils.conf_mat(
            f_labels, (f_preds > args.thres)).ravel()
        Tscore += np.array(tt)

        if num is not None and cnt >= num:
            break

    f1_mean = utils.fscore(Tscore)
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)


@torch.no_grad()
def test_tcn(dataset, model, args, iteration, device, logger=None, num=None, with_src=False):
    model.eval()

    Tscore = np.zeros(4)
    Tsrc = np.zeros(4)
    Tall = np.zeros(4)
    for cnt, ret in tqdm(
            enumerate(dataset.load_videos_all(is_training=False))):

        X, Y_forge, forge_time, Y_orig, gt_time, name = ret
        inputs = X.to(device)
        labels = torch.cat((Y_forge, Y_orig), 1).to(device)

        preds = model(inputs)
        preds = torch.sigmoid(preds)

        if with_src:
            pred_src = preds[:, 1]
            preds = preds[:, 0]

            labels_src = labels[:, 1]
            labels = labels[:, 0]

        f_preds = preds.squeeze().data.cpu().numpy().flatten()
        f_preds = f_preds > args.thres
        f_labels = labels.squeeze().data.cpu().numpy().flatten()
        f_labels = f_labels > 0.5

        tt = utils.conf_mat(
            f_labels, f_preds).ravel()
        Tscore += np.array(tt)

        if with_src:
            f_preds_s = pred_src.squeeze().data.cpu().numpy().flatten()
            f_preds_s = f_preds_s > args.thres
            f_labels_s = labels_src.squeeze().data.cpu().numpy().flatten()
            f_labels_s = f_labels_s > 0.5

            tt = utils.conf_mat(
                f_labels_s, f_preds_s).ravel()
            Tsrc += np.array(tt)

            f_label_all = (f_labels_s + f_labels).clip(max=1)
            f_pred_all = (f_preds + f_preds_s).clip(max=1)
            tt = utils.conf_mat(
                f_label_all, f_pred_all).ravel()
            Tall += np.array(tt)

        if num is not None and cnt >= num:
            break

    f1_mean = utils.fscore(Tscore)
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")

    if with_src:
        f1_mean_src = utils.fscore(Tsrc)
        print(f"F1 Score src: {f1_mean_src:.4f}")
        print(f"F1 Score all: {utils.fscore(Tall):.4f}\n")

    if logger is not None:
        logger.add_scalar("score/f1", f1_mean, iteration)
        if with_src:
            logger.add_scalar("score/f1_src", f1_mean_src, iteration)


@torch.no_grad()
def test_CMFD(dataset, args, root, num=None, logger=None):
    from scipy.io import loadmat

    Tscore = np.zeros(4)
    for cnt, ret in enumerate(tqdm(dataset.load_videos_all(to_tensor=False, shuffle=False))):
        X, Y_forge, forge_time, Y_orig, gt_time, name = ret

        matfile = os.papth.join(
            root, name+".mat"
        )

        assert os.path.exists(matfile)
        predY = loadmat(matfile)['map']
        predY = predY.transpose((2, 0, 1)).astype(np.float)

        for i in range(Y_forge.shape[0]):
            pred_o = predY[i]
            labels = (Y_forge[i] > 0) + (Y_orig[i] > 0)
            pred = skimage.transform.resize(pred_o, labels.shape[:2])
            # if np.sum(pred) > 0:
            #     print(np.max(pred))
            tt = utils.conf_mat(
                labels.ravel(), (pred > args.thres).ravel()).ravel()
            Tscore += np.array(tt)

    f1_mean = utils.fscore(Tscore)
    print()
    print("TEST")
    print(f"F1 Score: {f1_mean:.4f}")


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

        tt = utils.conf_mat(
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

        tforg = utils.conf_mat((labels == 2).ravel(),
                               preds_bool[:, 2].ravel()).ravel()

        tsrc = utils.conf_mat((labels == 1).ravel(),
                              preds_bool[:, 1].ravel()).ravel()

        tback = utils.conf_mat((labels == 0).ravel(),
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


@torch.no_grad()
def test_template_match(dataset, model, args, iteration, device,
                        logger=None, num=None):
    model.eval()
    iou_all = []
    Tscore = np.zeros(4)
    for i, ret in enumerate(dataset.load_data_template_match(is_training=False,
                                                             to_tensor=True, batch=True)):
        Xs, Xt, Ys = ret
        Xs, Xt, Ys = Xs.to(device), Xt.to(device), Ys.to(device)

        pred = model(Xs, Xt)

        gt_mask = Ys.data.cpu().numpy()
        pred_mask = torch.sigmoid(pred).data.cpu().numpy()

        f_gt = gt_mask > 0.5
        f_pred = pred_mask > args.thres
        iou = tools.iou_mask(f_gt, f_pred)
        print(f"\t{i}: {iou:.4f}")
        iou_all.append(iou)

        tt = utils.conf_mat(
            f_gt.ravel(), f_pred.ravel()).ravel()
        Tscore += np.array(tt)

        if num is not None and i >= num:
            break

    print(f"\nIoU: {np.mean(iou_all):.4f}")
    print(f"F1 source: {utils.fscore(Tscore)}")


@torch.no_grad()
def test_template_match_im(dataset, model, args, iteration, device,
                           logger=None, num=None):
    model.eval()
    iou_all_s = []
    iou_all_t = []
    # Tscore = np.zeros(4)
    for i, ret in enumerate(dataset.load_data_template_match_pair(is_training=False,
                                                                  to_tensor=True, batch=True)):
        Xs, Xt, Ys, Yt, name = ret
        Xs, Xt, Ys, Yt = Xs.to(device), Xt.to(device), Ys.to(device),\
            Yt.to(device)

        preds, predt = model(Xs, Xt)

        gt_mask_s = Ys.data.cpu().numpy()
        pred_mask_s = torch.sigmoid(preds).data.cpu().numpy()

        f_gt = gt_mask_s
        f_pred = pred_mask_s
        iou_s = tools.iou_mask_with_ignore(f_pred, f_gt,
                                           thres=args.thres)
        iou_all_s.append(iou_s)

        ####
        gt_mask_t = Yt.data.cpu().numpy()
        pred_mask_t = torch.sigmoid(predt).data.cpu().numpy()

        f_gt = gt_mask_t
        f_pred = pred_mask_t
        iou_t = tools.iou_mask_with_ignore(f_pred, f_gt,
                                           thres=args.thres)
        print(f"\t{i}: s: {iou_s:.4f}\t t:{iou_t:.4f}")
        iou_all_t.append(iou_t)

        # tt = utils.conf_mat(
        #     (f_gt>0.5).ravel(), f_pred.ravel()).ravel()
        # Tscore += np.array(tt)

        if num is not None and i >= num:
            break

    print(f"\nIoU_s: {np.mean(iou_all_s):.4f}")
    print(f"\nIoU_t: {np.mean(iou_all_t):.4f}")
    # print(f"F1 source: {utils.fscore(Tscore)}")
