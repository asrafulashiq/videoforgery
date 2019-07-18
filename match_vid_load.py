"""main file for training bubblenet type comparison patch matching
"""

import os
import numpy as np
import torch
from torch import nn
from torchvision import transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
import torch.nn.functional as F
import skimage
from tqdm import tqdm
# custom module
import config
from dataset import Dataset_image
from utils import CustomTransform, MultiPagePdf
from matching import tools, tools_argmax

from train import train_template_match_im
from test import test_template_match_im

import utils
import create_volume


def iou_time(t1, t2):
    iou = len(set(t1).intersection(set(t2)))/(
        len(set(t1).union(set(t2))) + 1e-8
    )
    return iou


def get_data(x, squeeze=True):
    if squeeze:
        x = x.squeeze()
    if x.is_cuda:
        x = x.data.cpu().numpy()
    else:
        x = x.data.numpy()
    return x


if __name__ == "__main__":
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    args = config.arg_main_im_match()
    print(args)
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    #! TEMPORARY
    args.videoset = 'tmp_youtube'

    # dataset
    tsfm = CustomTransform(size=args.size)

    dataset = Dataset_image(args=args, transform=tsfm)

    # * path to save
    root = Path("tmp_affinity")

    mask_processor = utils.Preprocessor(args)

    data_path = Path("./tmp_video_match")

    TTs = np.zeros(4)
    TTf = np.zeros(4)
    TTa = np.zeros(4)

    for fldr in tqdm(data_path.iterdir()):
        print(str(fldr).upper())
        if not fldr.is_dir():
            continue
        path_data = fldr / "data_pred.pt"
        Data = torch.load(str(path_data))
        # print("Data loaded from {}".format(path_data))

        X, Y_forge, Y_orig, gt_time, forge_time = Data['X'], \
            Data['Y_forge'], Data['Y_orig'], Data['gt_time'], \
            Data['forge_time']
        D_pred = Data['D_pred']
        name = Data['name']

        N = X.shape[0]
        path = root / name
        path.mkdir(parents=True, exist_ok=True)

        pdf = MultiPagePdf(total_im=N*N*2, out_name=str(path / "affinity.pdf"),
                           nrows=N, ncols=2, figsize=(5, N*2))

        Hist = np.zeros((N, N))

        D_np = np.zeros(tuple(D_pred.shape))

        for i in range(N):
            if i in forge_time:
                i_ind = np.where(forge_time == i)[0][0]
                gt_ind = gt_time[i_ind]
            else:
                gt_ind = None
            out1 = D_pred[i, :, 0]  # source
            out2 = D_pred[i, :, 1]  # forge
            out1 = out1.squeeze().data.cpu().numpy()
            out2 = out2.squeeze().data.cpu().numpy()

            for j in range(N):
                mask1 = out1[j]
                mask2 = out2[j]

                # preprocess mask
                mask1 = mask_processor.morph(mask1)
                mask2 = mask_processor.morph(mask2)

                if np.all(mask1 == 0):
                    mask2 = mask1
                if np.all(mask2 == 0):
                    mask1 = mask2

                rat = mask1.sum() * 1. / (mask2.sum() + 1e-8)
                rat = rat if rat > 1 else 1./rat
                if rat < 0.6:
                    mask1 = mask2 = np.zeros_like(mask1)

                D_np[i, j, 0] = mask1
                D_np[i, j, 1] = mask2

                im1 = tsfm.inverse(X[j])
                im2 = tsfm.inverse(X[i])
                im1_masked = im1 * mask1[..., None]
                im2_masked = im2 * mask2[..., None]

                # get histogram comparison
                vcomp = mask_processor.comp_hist(im1, im2, mask1, mask2,
                                                 compare_method=3)

                Hist[i, j] = 1 - vcomp

                ax = pdf.plot_one(im1_masked)
                ax.set_xlabel(f"{j}", fontsize="small")

                ax = pdf.plot_one(im2_masked)
                ax.set_xlabel(f"{i}", fontsize="small")

                ax.yaxis.set_label_position("right")
                ax.set_ylabel(f"Hist: {1-vcomp:.4f}")

                if gt_ind is not None and j == gt_ind:
                    ax.set_title("GT", fontsize="large")
        pdf.final()
        print("pdf saved")

        # matshow
        mat_gt = np.zeros((N, N))
        for _if, _ig in zip(forge_time, gt_time):
            mat_gt[_if, _ig] = 1.
        out_mat_name = str(path / "mat.pdf")

        create_volume.plot_conf_mat(mat_gt, str(path / "mat_gt.png"))
        create_volume.plot_conf_mat(Hist, str(path / "mat_pred.png"))

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(3, 3))
        axes[0].matshow(Hist)
        axes[1].matshow(mat_gt)
        fig.tight_layout()
        fig.savefig(out_mat_name)
        plt.close('all')

        detection
        det_arr = np.zeros(N)
        for k in range(N):
            det_arr[k] = np.mean(np.diag(Hist, k=-k))
        _ind = np.argmax(det_arr)
        pred_forge_time = np.arange(_ind, N)
        pred_gt_time = np.arange(0, len(pred_forge_time))

        # compare
        print("\tTime IoU: {}".format(iou_time(pred_forge_time, forge_time)))

        # get mask: 0:source, 1:forge
        Pred_mask_forge = np.zeros((N, *list(X.shape[-2:])))
        Pred_mask_src = np.zeros((N, *list(X.shape[-2:])))

        Pred_mask_src[pred_gt_time] = D_np[pred_forge_time, pred_gt_time, 0]
        Pred_mask_forge[pred_forge_time] = D_np[pred_forge_time,
                                                pred_gt_time, 1]
        Pred_mask_all = np.logical_or(Pred_mask_forge > 0.5, Pred_mask_src > 0.5)

        GT_forge = get_data(Y_forge)
        GT_src = get_data(Y_orig)
        GT_all = np.logical_or(GT_forge > 0.5, GT_src > 0.5)


        tforge = utils.conf_mat(
            GT_forge.ravel(), Pred_mask_forge.ravel()).ravel()
        tsrc = utils.conf_mat(
            GT_src.ravel(), Pred_mask_src.ravel()).ravel()
        tall = utils.conf_mat(
            GT_all.ravel(), Pred_mask_all.ravel()).ravel()

        TTs += tsrc
        TTf += tforge
        TTa += tall

        f_forge = utils.fscore(tforge)
        f_src = utils.fscore(tsrc)
        f_all = utils.fscore(tall)


        print()
        print("\t F_src : {:.4f}".format(f_src))
        print("\t F_forge : {:.4f}".format(f_forge))
        print("\t F_all : {:.4f}".format(f_all))

        # save all images

        folder_name = Path("tmp_tmp") / name

        folder_gt = folder_name / "gt"
        folder_pred = folder_name / "pred"
        folder_gt.mkdir(parents=True, exist_ok=True)
        folder_pred.mkdir(parents=True, exist_ok=True)

        for i_cnt in range(N):
            im = tsfm.inverse(X[i_cnt])
            if i_cnt != 0:
                im_with_gt = utils.add_overlay(im, GT_forge[i_cnt], c1=[1, 0, 0])
                im_with_pred = utils.add_overlay(
                    im, Pred_mask_forge[i_cnt], c1=[1, 0, 0]
                )

            else:
                im_with_gt = utils.add_overlay(im, GT_src[i_cnt])            
                im_with_pred = utils.add_overlay(
                    im, Pred_mask_src[i_cnt]
                )

            skimage.io.imsave(
                str(folder_gt / f"{i_cnt}.jpg"),
                skimage.img_as_ubyte(im_with_gt)
            )
            skimage.io.imsave(
                str(folder_pred / f"{i_cnt}.jpg"),
                skimage.img_as_ubyte(im_with_pred)
            )

        # plot volume
        create_volume.create(GT_src, GT_forge,
                             path=folder_name / "gt_vol.png")

        create_volume.create(Pred_mask_src, Pred_mask_forge,
                             path=folder_name / "pred_vol.png")

    print("FINAL Score:")
    print("Source : {:.4f}".format(utils.fscore(TTs)))
    print("Forge : {:.4f}".format(utils.fscore(TTf)))
    print("Mask : {:.4f}".format(utils.fscore(TTa)))
