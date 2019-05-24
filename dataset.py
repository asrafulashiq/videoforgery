from __future__ import print_function
from pathlib import Path
import os, sys
import cv2
import skimage
from skimage import io
import numpy as np
import torch

class Dataset_image:
    """class for dataset of image manipulation
    """

    def __init__(self, args=None, transform=None):
        # args contain necessary argument
        self.args = args
        self.videoset = args.videoset
        self.data_root = Path(args.root) / (self.videoset + "_tempered")
        self.data_root = self.data_root.expanduser()
        assert self.data_root.exists()

        self.transform = transform

        self.mask_root = self.data_root / "gt_mask"
        self.gt_root = self.data_root / "gt"
        self.im_mani_root = self.data_root / "vid"
        self._parse_all_images_with_gt()

    def split_train_test(self):
        ind = np.arange(len(self.data))
        np.random.shuffle(ind)
        ind_unto = int(len(self.data) * self.args.split)
        self.train_index = ind[:ind_unto]
        self.test_index = ind[ind_unto:]

    def _parse_all_images_with_gt(self):
        # self.__im_files_with_gt = []
        self.data = []
        for name in self.im_mani_root.iterdir():
            info = {"name": name.name, "files": []}
            for _file in name.iterdir():
                if _file.suffix == ".png":
                    im_file = str(_file)
                    mask_file = os.path.join(
                        str(self.mask_root), name.name, (_file.stem + ".jpg")
                    )
                    try:
                        assert os.path.exists(mask_file)
                    except AssertionError:
                        continue
                    info["files"].append((im_file, mask_file))
                    # self.__im_files_with_gt.append(
                    #     (im_file, mask_file)
                    # )
            self.data.append(info)
            self.split_train_test()

    def load_data(self, batch=10, is_training=True):
        counter = 1
        X = torch.empty((batch, 3, self.args.size, self.args.size), dtype=torch.float32)
        Y = torch.empty((batch, 1, self.args.size, self.args.size), dtype=torch.float32)

        for i, each_dat in enumerate(self.data):
            if is_training and i in self.train_index:
                # training mode
                for im_file, mask_file in each_dat["files"]:
                    image, mask = self.__get_im(im_file, mask_file)
                    X[counter] = image
                    Y[counter] = mask

                    if counter % batch == 0:
                        yield X, Y
                        X = torch.empty(
                            (batch, 3, self.args.size, self.args.size),
                            dtype=torch.float32,
                        )
                        Y = torch.empty(
                            (batch, 1, self.args.size, self.args.size),
                            dtype=torch.float32,
                        )
                    counter += 1
            else:  # testing mode
                pass

    # def __len__(self):
    #     return len(self.__im_files_with_gt)

    def __get_im(self, im_file, mask_file):
        image = io.imread(im_file)
        image = skimage.img_as_float32(image)  # image in [0-1] range

        _mask = io.imread(mask_file)

        if len(_mask.shape) > 2:
            mval = (0, 0, 255)
            ind = _mask[:, :, 2] > mval[2] / 2

            mask = np.zeros(_mask.shape[:2], dtype=np.float32)
            mask[ind] = 1
        else:
            mask = skimage.img_as_float32(_mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    def __getitem__(self, idx):
        im_file, mask_file = self.__im_files_with_gt[idx]
        image = io.imread(im_file)
        image = skimage.img_as_float32(image)  # image in [0-1] range

        _mask = io.imread(mask_file)

        if len(_mask.shape) > 2:
            mval = (0, 0, 255)
            ind = _mask[:, :, 2] > mval[2] / 2

            mask = np.zeros(_mask.shape[:2], dtype=np.float32)
            mask[ind] = 1
        else:
            mask = skimage.img_as_float32(_mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask, (im_file, mask_file)

    def get_frames_from_video(self):
        # randomly select one video and get frames (with labels)
        name = np.random.choice(list(self.im_mani_root.iterdir()))

        frame_list = []
        for _file in name.iterdir():
            if _file.suffix == ".png":
                im_file = str(_file)
                mask_file = os.path.join(
                    str(self.mask_root), name.name, (_file.stem + ".jpg")
                )
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue

            image = io.imread(im_file)
            image = skimage.img_as_float32(image)  # image in [0-1] range

            _mask = io.imread(mask_file)

            if len(_mask.shape) > 2:
                mval = (0, 0, 255)
                ind = _mask[:, :, 2] > mval[2] / 2

                mask = np.zeros(_mask.shape[:2], dtype=np.float32)
                mask[ind] = 1
            else:
                mask = skimage.img_as_float32(_mask)

            yield image, mask

