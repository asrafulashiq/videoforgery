from __future__ import print_function
from pathlib import Path
import os, sys
import cv2
import skimage
from skimage import io
import numpy as np


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


    def _parse_all_images_with_gt(self):
        self.__im_files_with_gt = []
        for name in self.im_mani_root.iterdir():
            for _file in name.iterdir():
                if _file.suffix == ".png":
                    im_file = str(_file)
                    mask_file = os.path.join(str(self.mask_root), name.name,
                                            (_file.stem+".jpg"))
                    try:
                        assert os.path.exists(mask_file)
                    except AssertionError:
                        continue
                    self.__im_files_with_gt.append(
                        (im_file, mask_file)
                    )

    def __len__(self):
        return len(self.__im_files_with_gt)

    def __getitem__(self, idx):
        im_file, mask_file = self.__im_files_with_gt[idx]
        image = io.imread(im_file)
        image = skimage.img_as_float32(image)  # image in [0-1] range

        _mask = io.imread(mask_file)

        if len(_mask.shape) > 2:
            mval = (0, 0, 255)
            ind = (_mask[:, :, 2] > mval[2]/2)

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
                mask_file = os.path.join(str(self.mask_root), name.name,
                                        (_file.stem+".jpg"))
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue

            image = io.imread(im_file)
            image = skimage.img_as_float32(image)  # image in [0-1] range

            _mask = io.imread(mask_file)

            if len(_mask.shape) > 2:
                mval = (0, 0, 255)
                ind = (_mask[:, :, 2] > mval[2]/2)

                mask = np.zeros(_mask.shape[:2], dtype=np.float32)
                mask[ind] = 1
            else:
                mask = skimage.img_as_float32(_mask)

            yield image, mask


