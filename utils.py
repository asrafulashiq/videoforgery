from scipy.ndimage import morphology
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
from skimage import transform

from scipy.ndimage import morphology
from scipy.ndimage import measurements

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def fscore(T_score):
    Fp, Fn, Tp = T_score[1:]
    f_score = 2 * Tp / (2 * Tp + Fp + Fn)
    return f_score


def conf_mat(labels, preds):
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels).ravel()
    if not isinstance(preds, np.ndarray):
        preds = np.array(preds).ravel()

    tp = np.sum(preds[labels==1]==1)
    tn = np.sum(preds[labels==0]==0)
    fp = np.sum(preds[labels==0]==1)
    fn = np.sum(preds[labels == 1] == 0)
    return np.array([tn, fp, fn, tp])


class SimTransform():
    def __init__(self, size=(224, 224)):
        if isinstance(size, int) or isinstance(size, float):
            size = (size, size)
        # scale
        self.scale = np.random.choice(
            np.linspace(0.9, 1.1, 30)
        )
        # rotation
        self.rot = np.random.choice(
            np.linspace(-np.pi/10, np.pi/10, 50)
        )
        # translate
        self.translate = (
            np.random.choice(np.linspace(-0.05 * size[1], 0.1 * size[1], 50)),
            np.random.choice(np.linspace(-0.05 * size[0], 0.1 * size[0], 50))
        )

        # flip lr
        self.flip = np.random.rand() > 0.5

        self.tfm = transform.SimilarityTransform(
            scale=self.scale,
            translation=self.translate)

    def __call__(self, im=None, mask=None):
        if im is not None:
            im = transform.warp(im, self.tfm)
            # im = transform.rotate(im, self.rot)
            if self.flip:
                im = np.flip(im, 1).copy()
        if mask is not None:
            mask = transform.warp(mask, self.tfm)
            # mask = transform.rotate(mask, self.rot)
            if self.flip:
                mask = np.flip(mask, 1).copy()
        return im, mask



class CustomTransform:
    def __init__(self, size=224):
        if isinstance(size, int) or isinstance(size, float):
            self.size = (size, size)
        else:
            self.size = size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def resize(self, img=None, mask=None):
        if img is not None:
            img = skimage.img_as_float32(img)
            if img.shape[0] != self.size[0] or img.shape[1] != self.size[1]:
                img = cv2.resize(
                    img, self.size, interpolation=cv2.INTER_LINEAR)

        if mask is not None:
            if mask.shape[0] != self.size[0] or mask.shape[1] != self.size[1]:
                mask = cv2.resize(
                    mask, self.size, interpolation=cv2.INTER_NEAREST)
        return img, mask

    def inverse(self, x):
        if x.is_cuda:
            x = x.squeeze().data.cpu().numpy()
        else:
            x = x.squeeze().data.numpy()
        if len(x.shape) < 3:
            pass
        else:
            x = x.transpose((1, 2, 0))
            x = x * self.std + self.mean
        return x


    def __call__(self, img=None, mask=None, other_tfm=None):
        img, mask = self.resize(img, mask)
        if other_tfm is not None:
            img, mask = other_tfm(img, mask)
        if img is not None:
            img = (img - self.mean) / self.std
            img = self.to_tensor(img)

        if mask is not None:
            mask = self.to_tensor(mask)

            return img, mask

        else:
            return img


def custom_transform_images(images=None, masks=None, size=224, other_tfm=None):
    tsfm = CustomTransform(size=size)
    X, Y = None, None
    if images is not None:
        X = torch.zeros((images.shape[0], 3, size, size), dtype=torch.float32)
        for i in range(images.shape[0]):
            X[i] = tsfm(img=images[i], other_tfm=other_tfm)
    if masks is not None:
        Y = torch.zeros((masks.shape[0], 1, size, size), dtype=torch.float32)
        for i in range(masks.shape[0]):
            _, Y[i, 0] = tsfm(img=None, mask=masks[i], other_tfm=other_tfm)

    return X, Y


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
        if len(mask.shape) > 2:
            mask = skimage.color.rgb2gray(mask) > 0
        xx, yy = np.nonzero(mask.squeeze())
        x1, x2, y1, y2 = np.min(xx), np.max(xx), np.min(yy), np.max(yy)
        mask = np.ones(im.shape[:2], dtype=im.dtype)
        mask[x1:x2, y1:y2] = 0
        im_masked = im * mask[..., None]
    elif type == "foreground-bbox":
        if len(mask.shape) > 2:
            mask = skimage.color.rgb2gray(mask) > 0
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


def patch_transform(im_mask, mask_bb, new_centroid, translate=None, scale=None):
    if len(mask_bb) < 4:
        return im_mask.copy()

    patch_mask = im_mask[mask_bb[1]:mask_bb[3], mask_bb[0]:mask_bb[2]]
    resized_patch = cv2.resize(patch_mask, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_NEAREST)

    hp, wp = resized_patch.shape[:2]

    topx = int(max(0, new_centroid[0] - wp/2))
    topy = int(max(0, new_centroid[1] - hp/2))

    bottomx = int(min(topx+wp, im_mask.shape[1]))
    bottomy = int(min(topy+hp, im_mask.shape[0]))

    w, h = bottomx - topx, bottomy - topy

    new_mask = np.zeros(im_mask.shape, dtype=patch_mask.dtype)

    new_mask[topy:topy+h, topx:topx+w] = resized_patch[:h, :w]

    return new_mask


def splice(img_target, img_source, img_mask, do_blend=False):

    if img_target.shape != img_source.shape:
        img_target = skimage.transform.resize(
            img_target, img_mask.shape[:2], anti_aliasing=True, mode='reflect'
        )
        # img_target = skimage.img_as_ubyte(img_target)

    if len(img_mask.shape) < 3:
        img_mask = img_mask[..., None]

    img_mask = (img_mask > 0)
    if img_mask.dtype != np.float32:
        img_mask = img_mask.astype(np.float32)
    img_mani = img_mask * img_source + img_target * (1 - img_mask)
    # img_mani = img_mani.astype(np.uint8)

    return img_mani


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


def add_overlay(im, m1, m2=None, alpha=0.5, c1=[0, 1, 0], c2=[1, 0, 0]):
    r, c = im.shape[:2]

    M1 = np.zeros((r, c, 3), dtype=np.float32)
    M2 = np.zeros((r, c, 3), dtype=np.float32)

    if m2 is not None:
        M1[m1 > 0] = c1
        M2[m2 > 0] = c2
        M = cv2.addWeighted(M1, alpha, M2, 1 - alpha, 0, None)
    else:
        M1[m1 > 0] = c1
        M = M1

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
    def __init__(self, range=(0.2, 1.5), thres=0.5):
        self.scale_range = np.linspace(*range, 30)
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
        for scale in self.scale_range[::-1]:

            # if (template.shape[0]*scale) < 5 or (template.shape[1]*scale) < 5:
            #     continue

            # template_resized = cv2.resize(template, None, fx=scale, fy=scale)
            # r = scale
            im_resized = skimage.img_as_ubyte(
                cv2.resize(image, None, fx=scale, fy=scale))

            # if the resized image is smaller than the template, then break
            # from the loop
            # if iH < template_resized.shape[0] or iW < template_resized.shape[1]:
            #     break
            if im_resized.shape[0] < template.shape[0] or \
                    im_resized.shape[1] < template.shape[1]:
                break

            im_edged = cv2.Canny(im_resized, 50, 200)
            temp_edged = cv2.Canny(skimage.img_as_ubyte(template), 50, 200)

            # result = cv2.matchTemplate(image, template_resized,
            #                             cv2.TM_CCOEFF_NORMED)
            result = cv2.matchTemplate(
                im_edged,
                temp_edged,
                cv2.TM_CCOEFF_NORMED
            )
            (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

            # if we have found a new maximum correlation value, then ipdate
            # the bookkeeping variable
            if found is None or maxVal > found[0]:
                found = (maxVal, maxLoc, scale)
                # print("----------")
                # print("max val : ", maxVal, " r :", 1 / r)

        # unpack the bookkeeping varaible and compute the (x, y) coordinates
        # of the bounding box based on the resized ratio
        (_, maxLoc, r) = found
        # w, h = int(template.shape[1] * r), int(template.shape[0] * r)
        # x, y = maxLoc[:2]

        # # cv2.rectangle(image, (x, y), (x + w - 1, y + h - 1), (1.0, 0.0, 0.0), 4)

        # t_res = cv2.resize(template, None, fx=r, fy=r)
        # nim = np.zeros(image.shape[:2])
        # nim[y: y + t_res.shape[0], x : x+t_res.shape[1]] = (t_res > 0.5)

        x, y = int(maxLoc[0] / r), int(maxLoc[1] / r)
        nim = np.zeros(image.shape[:2])
        tmp_resized = cv2.resize(template, None, fx=1/r, fy=1/r)
        h, w = tmp_resized.shape[:2]
        nim[y: y + tmp_resized.shape[0], x: x +
            tmp_resized.shape[1]] = (tmp_resized > 0.5)

        return (x, y, w, h), found[0], nim


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
        self.figsize = tuple(figsize)
        self.out_name = out_name

        # create figure and axes
        total_pages = int(np.ceil(total_im / (nrows * ncols)))

        self.figs = []
        self.axes = []

        for _ in range(total_pages):
            f, a = plt.subplots(nrows, ncols)

            f.set_size_inches(figsize)
            self.figs.append(f)
            self.axes.extend(a.flatten())

        self.cnt_ax = 0

    def plot_one(self, x, *args, **kwargs):
        ax = self.axes[self.cnt_ax]
        ax.imshow(x, *args, **kwargs)  # prediction
        # ax.imshow(x[0])  # ground truth

        ax.set_xticks([])
        ax.set_yticks([])

        self.cnt_ax += 1
        return ax

    def final(self):
        with PdfPages(self.out_name) as pdf:
            for fig in self.figs:
                fig.tight_layout()
                pdf.savefig(fig)
        plt.close("all")


class Preprocessor():
    def __init__(self, args):
        self.args = args

    # morphological preprocessing
    def morph(self, mask):
        # threshold
        mask = mask > self.args.thres

        # do closing
        mask_closed = morphology.binary_closing(mask, structure=np.ones((5, 5)))

        # do opening
        mask_opened = morphology.binary_closing(
            mask_closed, structure=np.ones((5, 5)))

        # hole filling
        mask_filled = morphology.binary_fill_holes(
            mask_opened, structure=np.ones((8, 8)))

        # get maximum connected component
        labels, _ = measurements.label(mask_filled)
        if len(np.unique(labels)) == 1:
            return np.zeros_like(mask)

        mask_maxlab = labels == np.argmax(np.bincount(labels.flat)[1:])+1

        return mask_maxlab.astype(np.float)

    def calc_hist(self, im, mask=None):
        # convert to hsv color space
        im = skimage.img_as_ubyte(im)
        mask = skimage.img_as_ubyte(mask)
        im_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)

        h_bins = 50
        s_bins = 50
        histSize = [h_bins, s_bins]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = h_ranges + s_ranges  # concat lists
        # Use the 0-th and 1-st channels
        channels = [0, 1]

        hist_base = cv2.calcHist([im_hsv], channels, mask,
                                 histSize, ranges, accumulate=False)
        cv2.normalize(hist_base, hist_base, alpha=0,
                     beta=1, norm_type=cv2.NORM_MINMAX)
        return hist_base

    def comp_hist(self, x1, x2, mask1=None, mask2=None, compare_method=0):
        hist1 = self.calc_hist(x1, mask1)
        hist2 = self.calc_hist(x2, mask2)

        value = cv2.compareHist(hist1, hist2, compare_method)
        return value
