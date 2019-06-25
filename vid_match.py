from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage
import os
from tqdm import tqdm
import numpy as np

from sklearn import metrics
import cv2
import utils
from utils import MultiPagePdf, add_overlay
from utils import CustomTransform


def get_patch(im):
    # im: image with patch
    x, y = np.nonzero(im)
    x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
    patch = im[x1:x2, y1:y2]
    return patch


def template_vid(X_m, X_ref, matcher, Y_orig, act_ind):

    if X_ref.size == 0:
        return []

    X = np.zeros_like(X_m)
    for i in range(X.shape[0]):
        if i in act_ind:
            ind = act_ind.index(i)
            X[i] = utils.image_with_mask(X_m[i], X_ref[ind],
                        'background-bbox')
        else:
            X[i] = X_m[i]

    sim_list = np.zeros((X_ref.shape[0], X.shape[0]), dtype=np.float)
    bb_list = np.zeros((X_ref.shape[0], X.shape[0], 4), dtype=np.int)

    for i in range(X_ref.shape[0]):
        for j in range(X.shape[0]):
            out = matcher.match(X[j], X_ref[i])
            bbox, val, patched_im = out
            sim_list[i, j] = val
            bb_list[i, j] = bbox
            # _tinfo = metrics.confusion_matrix(
            #     Y_orig[i].ravel(), patched_im.ravel()
            # ).ravel()
            # T_score += np.array(_tinfo)

    gt_len = X_ref.shape[0]

    arr0 = np.arange(gt_len)
    flag = [-1, -1]
    for k in range(0, X.shape[0]-gt_len+1):
        arr = arr0 + k
        val = sim_list[arr0, arr]
        prod = np.prod(val)

        if prod > flag[0]:
            flag = (prod, k)

    bb = bb_list[arr0, arr0+flag[1]]

    #

    T_score = np.zeros(4)

    X_src_mask = np.zeros((gt_len,X.shape[1], X.shape[2]))
    for i in range(X_ref.shape[0]):
        ref = X_ref[i]
        patch = get_patch(skimage.color.rgb2gray(ref))
        x, y, w, h = bb[i]
        patch_res = cv2.resize(patch, (w, h), interpolation=cv2.INTER_NEAREST)
        nim = np.zeros((X_ref.shape[1], X_ref.shape[2]))
        nim[y:y+h, x:x+w] = (patch_res > 0)
        X_src_mask[i] = nim

        # compare
        _tinfo = metrics.confusion_matrix(
            Y_orig[i].ravel(), nim.ravel()
        ).ravel()
        T_score += np.array(_tinfo)

    X_src = np.zeros(X_m.shape[:3])
    X_src[flag[1]: flag[1]+gt_len] = X_src_mask

    return (flag[1], flag[1]+gt_len), T_score, X_src


def match_in_the_video(dataset, args):
    """match an image with forged match in the video
    """

    matcher = utils.TemplateMatch(thres=args.thres)

    list_top = []
    list_iou = []
    for cnt, tmp in tqdm(enumerate(dataset.get_search_from_video(first_only=False))):
        X, X_ref, gt_ind, first_ind = tmp

        sim_list = np.zeros((X_ref.shape[0], X.shape[0]), dtype=np.float)
        for i in range(X_ref.shape[0]):
            for j in range(X.shape[0]):
                out = matcher.match(X[j], X_ref[i])
                bbox, val, patched_im = out
                sim_list[i, j] = val

        sort_ind = np.argsort(-sim_list, axis=1)

        for k in range(X_ref.shape[0]):
            tmp = np.where(sort_ind[k] == gt_ind)[0][0]+1
            list_top.append(tmp)

        gt_len = X_ref.shape[0]

        arr0 = np.arange(gt_len)
        flag = [-1, -1]
        for k in range(0, X.shape[0]-gt_len):
            arr = arr0 + k
            val = sim_list[arr0, arr]
            prod = np.prod(val)

            if prod > flag[0]:
                flag = (prod, k)

        # calculate iou
        gt = set(np.arange(gt_ind, gt_ind+gt_len+1))
        pred = set(np.arange(flag[1], flag[1]+gt_len+1))

        iou = len(gt.intersection(pred)) / len(gt.union(pred))
        list_iou.append(iou)

        print("#{} - iou : {:.2f}".format(cnt, iou))

    # top-k score
    print("-----------------------")

    for fl in range(1, 6):
        sc = [1 if i <= fl else 0 for i in list_top]
        print("Top-{} accuracy: {:.2f}".format(
            fl, np.sum(sc) / len(sc)
        ))

    # iou
    print("IoU mean : ", np.mean(list_iou))