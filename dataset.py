from __future__ import print_function
from pathlib import Path
import os, sys
import pickle
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
        self.__im_files_with_gt = []
        self.data = []
        for i, name in enumerate(self.im_mani_root.iterdir()):
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
                    self.__im_files_with_gt.append(
                        (i, im_file, mask_file)
                    )
            self.data.append(info)
            self.split_train_test()

    def load_data(self, batch=20, is_training=True, shuffle=True):
        counter = 0
        X = torch.empty((batch, 3, self.args.size, self.args.size), dtype=torch.float32)
        Y = torch.empty((batch, 1, self.args.size, self.args.size), dtype=torch.float32)
        Info = []

        if shuffle:
            np.random.shuffle(self.__im_files_with_gt)

        for i, im_file, mask_file in self.__im_files_with_gt:
            if (is_training and i in self.train_index) or \
                (not is_training and i in self.test_index):
                image, mask = self.__get_im(im_file, mask_file)
                X[counter] = image
                Y[counter] = mask
                Info.append((im_file, mask_file))
                counter += 1
                if counter % batch == 0:
                    yield X, Y, Info
                    X = torch.empty(
                        (batch, 3, self.args.size, self.args.size),
                        dtype=torch.float32,
                    )
                    Y = torch.empty(
                        (batch, 1, self.args.size, self.args.size),
                        dtype=torch.float32,
                    )
                    Info = []
                    counter = 0
        if counter != 0:
            yield X, Y, Info

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


    def load_triplet(self, num=10):
        # randomly select one video and get frames (with labels)
        name = np.random.choice(list(self.im_mani_root.iterdir()))
        gt_file = os.path.join(
            str(self.gt_root), name.name + ".pkl"
        )

        with open(gt_file, "rb") as fp:
            dat = pickle.load(fp)

        filenames = list(dat.keys())

        list_forged_ind = []  # filename indices having forged part
        for i, f in enumerate(filenames):
            if dat[f]["mask_orig"] is not None:
                list_forged_ind.append(i)

        for i in range(num):
            ind = np.random.choice(list_forged_ind)
            cur_file = filenames[ind]
            cur_data = dat[cur_file]

            mask_orig = cur_data["mask_orig"]
            mask_new = cur_data["mask_new"]
            offset = cur_data["offset"]
            src_file = filenames[ind - offset]


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




    def get_frames_from_video(self):
        # randomly select one video and get frames (with labels)
        name = np.random.choice(list(self.im_mani_root.iterdir()))
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

