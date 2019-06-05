from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11
from torchvision import transforms
import numpy as np
import cv2
import skimage
import imutils
from skimage import io

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class CustomTransform:
    def __init__(self, size=224):
        self.size = (size, size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask=None):
        if img is not None:
            if img.dtype == np.uint8:
                img = (img / 255.0).astype(np.float32)

            img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
            img = (img - self.mean) / self.std
            img = self.to_tensor(img)

        if mask is not None:
            mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
            mask = self.to_tensor(mask)

            return img, mask

        else:
            return img


def image_with_mask(im, mask, type="foreground", blend=False):
    im = skimage.img_as_float32(im)
    mask = skimage.img_as_float32(mask)

    if len(im.shape) > len(mask.shape):
        mask = mask[..., None]

    if type == "foreground":
        im_masked = im * mask
    elif type == "background":
        im_masked = im * (1 - mask)
    elif type == "background-bbox":
        xx, yy = np.nonzero(mask.squeeze())
        x1, x2, y1, y2 = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        mask = np.ones(im.shape[:2], dtype=im.dtype)
        mask[x1:x2, y1:y2] = 0
        im_masked = im * mask[..., None]
    elif type == "foreground-bbox":
        xx, yy = np.nonzero(mask.squeeze())
        x1, x2, y1, y2 = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        mask = np.zeros(im.shape[:2], dtype=im.dtype)
        mask[x1:x2, y1:y2] = 1
        im_masked = im * mask[..., None]
    if blend:
        blend_ratio = 0.3
        im_masked = cv2.addWeighted(
            im, blend_ratio, im_masked, 1 - blend_ratio, 0, None
        )
    return im_masked


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

    M = cv2.addWeighted(M1, alpha, M2, 1 - alpha, 0, None)

    I = cv2.addWeighted(im, alpha, M, 1 - alpha, 0, None)

    return I


def iou_time(gt, pred):
    # calculate iou
    gt = set(np.arange(gt[0], gt[1] + 1))

    if pred:
        pred = set(np.arange(pred[0], pred[1] + 1))
    else:
        pred = set([])
    iou = len(gt.intersection(pred)) / len(gt.union(pred))
    return iou


class TemplateMatch:
    def __init__(self, range=(0.7, 1.3), thres=0.5):
        self.scale_range = np.linspace(*range, 20)
        self.thres = thres

    def get_patch(self, im):
        # im: image with patch
        x, y = np.nonzero(im)
        x1, x2, y1, y2 = np.min(x), np.max(x), np.min(y), np.max(y)
        patch = im[x1:x2, y1:y2]
        return patch

    def match(self, image, im_template):
        # convert image to gray
        image = skimage.color.rgb2gray(image)
        im_template = skimage.color.rgb2gray(im_template)
        template = self.get_patch(im_template)

        found = None
        (iH, iW) = image.shape[:2]

        # loop over the scales of the image
        for scale in self.scale_range:

            template_resized = cv2.resize(
                template, None, fx=scale, fy=scale)
            r = scale

            # if the resized image is smaller than the template, then break
            # from the loop
            if iH < template_resized.shape[0] or iW < template_resized.shape[1]:
                break

            result = cv2.matchTemplate(image, template_resized, cv2.TM_CCOEFF_NORMED)
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, r)
                # print("----------")
                # print("max val : ", maxVal, " r :", 1 / r)

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        w, h = int(template.shape[1] * r), int(template.shape[0] * r)
        x, y = maxLoc[:2]

        cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (1.0, 0.0, 0.0), 4)

        return (x, y, w, h), found[0], image


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
        total_pages = int(np.ceil(total_im / (nrows * ncols)))

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
        plt.close("all")
