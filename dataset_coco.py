from pathlib import Path
import os, sys
import pickle
import cv2
import skimage
from skimage import io
import numpy as np
import torch
from torchvision import transforms

from pycocotools.coco import COCO

import utils
import config

class COCODataset:
    def __init__(self, args=None, transform=None):
        self.dataDir = Path('~/dataset/coco').expanduser()
        self.year = '2014'
        self.train_ann_file = self.dataDir / 'annotations' / \
            'instances_train{}.json'.format(self.year)
        self.test_ann_file = self.dataDir / 'annotations' / \
            'instances_val{}.json'.format(self.year)
        self.train_im_folder = self.dataDir / 'images' / f'train{self.year}'
        self.test_im_folder = self.dataDir / 'images' / f'val{self.year}'

        self.args = args

        if transform is None:
            self.transform = transforms.ToTensor
        else:
            self.transform = transform

    def load_data(self, batch=20, is_training=True,
                  shuffle=True, with_boundary=False):
        if is_training:
            annFile = self.train_ann_file
            imDir = self.train_im_folder
        else:
            annFile = self.test_ann_file
            imDir = self.test_im_folder

        counter = 0
        X = torch.zeros((batch, 3, self.args.size, self.args.size), dtype=torch.float32)
        if with_boundary:
            ysize = 2
        else:
            ysize = 1
        Y = torch.zeros((batch, ysize, self.args.size, self.args.size),
                        dtype=torch.float32)

        coco = COCO(annFile)
        imids = coco.getImgIds()

        if shuffle:
            np.random.shuffle(imids)
        Info = []

        for _id in imids:
            id_dest = _id
            id_source = np.random.choice(imids)

            try:
                image, mask = self.blend_ims(id_source, id_dest, coco, imDir)

                image, mask = self.transform(image, mask)
                X[counter] = image
                Y[counter, 0] = mask
                # if with_boundary:
                #     Y[counter, 1] = tmp[2]
                Info.append((id_source, id_dest))
            except cv2.error:
                continue
            counter += 1
            if counter % batch == 0:
                if torch.any(torch.isnan(Y)):
                    import pdb

                    pdb.set_trace()
                yield X, Y, Info

                if torch.any(Y < 0):
                    import pdb
                    pdb.set_trace()
                X = torch.zeros(
                    (batch, 3, self.args.size, self.args.size), dtype=torch.float32
                )
                Y = torch.zeros(
                    (batch, ysize, self.args.size,
                        self.args.size), dtype=torch.float32
                )
                Info = []
                counter = 0

    def blend_ims(self, id1, id2, coco, imDir):
        im1, mask1 = self.__get_im(id1, coco, imDir)
        im2, mask2 = self.__get_im(id2, coco, imDir)

        # get random centroid, translate, scale
        centroid = np.array([np.random.choice(im1.shape[1]),
                            np.random.choice(im1.shape[0])])
        centroid_orig, mask_orig_bb = self.get_centroid_from_mask(mask1)

        if centroid_orig is None:
            return im2, mask1

        translate = centroid - centroid_orig
        scale = np.random.choice(np.linspace(0.9, 2.4, 20))

        im_mask_new = utils.patch_transform(mask1, mask_orig_bb, centroid,
                                            translate, scale)
        im_s_masked = mask1[..., None] * im1

        im_s_n = utils.patch_transform(im_s_masked, mask_orig_bb,
                                        centroid, translate, scale)
        im_mani = utils.splice(im2, im_s_n, im_mask_new)
        return im_mani, im_mask_new

    @staticmethod
    def get_centroid_from_mask(mask):
        m_y, m_x = np.where(mask > 0)

        if m_y.size == 0:
            return None, None
        x1, x2 = np.min(m_x), np.max(m_x)
        y1, y2 = np.min(m_y), np.max(m_y)

        mask_orig_bb = (x1, y1, x2, y2)

        cent = np.array([int((x1+x2)/2), int((y1+y2)/2)])
        return cent, mask_orig_bb

    def __get_im(self, imid, coco, imDir):
        im_info = coco.loadImgs([imid])[0]
        img = skimage.img_as_float32(io.imread(
            str(imDir / im_info['file_name'])
        ))
        img = cv2.resize(img, (self.args.size, self.args.size))
        if len(img.shape) < 3:
            img = skimage.color.gray2rgb(img)
        try:
            anns = coco.getAnnIds(imgIds=imid)
            if not anns:
                raise ValueError
            np.random.shuffle(anns)
            for annid in anns:
                ann = coco.loadAnns([annid])[0]
                mask = coco.annToMask(ann)
                mask = cv2.resize(mask, (self.args.size, self.args.size))
                mask = (mask > 0).astype(np.float32)
                if np.sum(mask) / (mask.shape[0] * mask.shape[1]) < 0.02:
                    continue
        except ValueError:
            mask = np.zeros(img.shape[:2], dtype=np.float32)

        return img, mask

if __name__ == '__main__':

    from utils import CustomTransform
    args = config.arg_main()
    np.random.seed(args.seed)
    tsfm = CustomTransform(size=args.size)
    dataset = COCODataset(args=args, transform=tsfm)
    loader = dataset.load_data(is_training=False)
    img, mask, _ = next(loader)

    print(img.shape)
    print(mask.shape)
