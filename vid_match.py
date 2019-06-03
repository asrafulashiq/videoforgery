from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import skimage
import os
from tqdm import tqdm
import numpy as np

# metric
from sklearn import metrics

import utils
from utils import MultiPagePdf, add_overlay
from utils import CustomTransform

import warnings
warnings.filterwarnings("ignore")


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