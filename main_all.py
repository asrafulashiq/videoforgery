"""The full program, from forgery detection to matching is here
"""


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
from sklearn import metrics

# custom module
from models import Model
import config
from dataset import Dataset_image
from utils import CustomTransform
import utils
from train import train
from test import test
import vid_match


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

    T_score_cum = np.zeros(4)
    T_score_cum_copy = np.zeros(4)

    for cnt, dat in (enumerate(dataset.load_videos_all())):
        # work with a particular video
        X, Y_forge, forge_time, Y_orig, gt_time, vid_name = dat

        Pred = np.zeros(X.shape[:3], dtype=np.float32)
        Pred_im = np.zeros(X.shape[:4], dtype=np.float32)

        act_ind = []

        T_score = np.zeros(4) # tn, fp, fn, tp

        for i in (range(X.shape[0])):
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
                area = [(_i, np.sum(pred_lab==_i)) for _i in labels if _i != 0]
                max_lab, _ = max(area, key = lambda x: x[1])
                pred_lab[pred_lab != max_lab] = 0
                pred_lab[pred_lab == max_lab] = 1
                pred_lab = pred_lab.astype(np.float)

                if np.sum(pred_lab) / (args.size**2) < 0.02:
                    pred_lab = np.zeros_like(pred_lab)
                else:
                    act_ind.append(i)

            Pred[i] = pred_lab
            Pred_im[i] = utils.image_with_mask(X[i], pred_lab)

            _tinfo = metrics.confusion_matrix(
                Y_forge[i].ravel(), pred_lab.ravel()
            ).ravel()
            T_score += np.array(_tinfo)


            #! TEST
            # pp = utils.image_with_mask(X[i], pred_lab, type="foreground", blend=True)
            # path = Path(f"tmp_all/{cnt}")
            # path.mkdir(parents=True, exist_ok=True)
            # io.imsave(str(path/f"{i}.jpg"), skimage.img_as_ubyte(pp))


        T_score_cum += T_score

        f_score = utils.fscore(T_score)

        # match
        matcher = utils.TemplateMatch(thres=args.thres)

        if forge_time is None or len(act_ind)==0:
            continue

        X_ref = Pred_im[act_ind]

        pred_t, tscore, Y_orig_pred = vid_match.template_vid(X, X_ref, matcher,
                                                            Y_orig, act_ind)

        T_score_cum_copy += tscore
        f_copy = utils.fscore(tscore)

        iou_copy = utils.iou_time(gt_time, pred_t)

        if act_ind:
            strt = act_ind[0]
            if len(act_ind) == 1:
                end = X.shape[0]
            else:
                end = act_ind[-1]
            pred_forge_time = (strt, end)

        iou_move = utils.iou_time(forge_time, pred_forge_time)

        print(f"{cnt:2d} {vid_name:>15s}: IoU -  move: {iou_move:.2f}  copy: {iou_copy:.2f}"
        + f"  F1: {f_score:.2f}" + f"  F1-copy: {f_copy:.2f}")

        for i in range(X.shape[0]):
            im = X[i]
            mask_forge = Pred[i]
            mask_orig = Y_orig_pred[i]

            image = utils.add_overlay(im, mask_orig, mask_forge)

            # pp = utils.image_with_mask(X[i], pred_lab, type="foreground", blend=True)
            path = Path(f"tmp_all/{vid_name}")
            path.mkdir(parents=True, exist_ok=True)
            io.imsave(str(path/f"{i}.jpg"), skimage.img_as_ubyte(image))


    f_score_cum = utils.fscore(T_score_cum)

    print("--------------")
    print(f" F1 (cumulative) : {f_score_cum:.2f}")
    print(f" F1-copy (cumulative) : {utils.fscore(T_score_cum_copy):.2f}")
