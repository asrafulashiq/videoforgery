from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11
from torchvision import transforms
import numpy as np
import cv2

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class CustomTransform():
    def __init__(self, size=224):
        self.size = (size, size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask=None):
        if img is not None:
            if img.dtype == np.uint8:
                img = (img / 255.).astype(np.float32)

            img = cv2.resize(img, self.size, 0, 0, cv2.INTER_LINEAR)
            img = (img - self.mean) / self.std
            img = self.to_tensor(img)

        if mask is not None:
            mask = cv2.resize(mask, self.size)
            mask = self.to_tensor(mask)

            return img, mask

        else:
            return img


def overlay_masks(m1, m2, alpha=0.5):
    """overlay two boolean mask

    Arguments:
        m1 {boolean array} -- original mask
        m2 {boolean array} -- predicted mask
    """
    r, c = m1.shape[:2]
    M1 = np.zeros((r, c, 4), dtype=np.float)
    M2 = np.zeros((r, c, 4), dtype=np.float)

    M1[m1 > 0] = [0, 1, 0, 0.5]
    M2[m2 > 0] = [1, 0, 0, 0.5]

    return (M1, M2)

def add_overlay(im, m1, m2, alpha=0.5):
    r, c = im.shape[:2]

    M1 = np.zeros((r, c, 3), dtype=np.float32)
    M2 = np.zeros((r, c, 3), dtype=np.float32)

    M1[m1 > 0] = [0, 1, 0]
    M2[m2 > 0] = [1, 0, 0]

    M = cv2.addWeighted(M1, alpha, M2, 1-alpha, 0, None)

    I = cv2.addWeighted(im, alpha, M, 1-alpha, 0, None)

    return I



class MultiPagePdf:
    def __init__(self, total_im, out_name, nrows=4, ncols=4, figsize=(8, 6)):
        """init

        Keyword Arguments:
            total_im {int} -- #images
            nrows {int} -- #rows per page (default: {4})
            ncols {int} -- #columns per page (default: {4})
            figsize {tuple} -- fig size (default: {(8, 6)})
        """
        self.total_im = total_im
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.out_name = out_name

        # create figure and axes
        total_pages = int(np.ceil(total_im/(nrows*ncols)))

        self.figs = []
        self.axes = []

        for i in range(total_pages):
            f, a = plt.subplots(nrows, ncols)

            f.set_size_inches(figsize)
            self.figs.append(f)
            self.axes.extend(a.flatten())

        self.cnt_ax = 0

    def plot_one(self, x):
        ax = self.axes[self.cnt_ax]
        ax.imshow(x)  # prediction
        # ax.imshow(x[0])  # ground truth

        ax.set_xticks([])
        ax.set_yticks([])

        self.cnt_ax += 1

    def final(self):
        with PdfPages(self.out_name) as pdf:
            for fig in self.figs:
                pdf.savefig(fig)
        plt.close('all')
