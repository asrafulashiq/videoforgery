from __future__ import print_function
from pathlib import Path
import os, sys
import pickle
import cv2
import skimage
from skimage import io
import numpy as np
import torch
from torchvision import transforms



def add_sorp(im, type="pepper"):
    im = skimage.img_as_float32(im)
    if type == "pepper":
        mask = np.ones(im.shape[:2], dtype=np.float32)
        mask = skimage.util.random_noise(mask, mode=type)
        if mask.shape != im.shape:
            mask = mask.reshape(im.shape)
        im = im * mask
    elif type == "salt":
        mask = np.zeros(im.shape[:2], dtype=np.float32)
        mask = skimage.util.random_noise(mask, mode=type)
        im = im.copy()
        if len(im.shape) > 2:
            im[mask > 0] = (1, 1, 1)
        else:
            im[mask > 0] = 1

    return im



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

        if transform is None:
            self.transform = transforms.ToTensor()
        else:
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
                        str(self.mask_root), name.name, (_file.stem + ".png")
                    )
                    try:
                        assert os.path.exists(mask_file)
                    except AssertionError:
                        continue
                    info["files"].append((im_file, mask_file))
                    self.__im_files_with_gt.append((i, im_file, mask_file))
            self.data.append(info)
            self.split_train_test()

    def randomize_mask(self, im):
        rand = np.random.randint(1, 3)
        kernel = np.ones((5, 5))
        if rand == 1:  # erosion
            im = cv2.erode(im, kernel)
        elif rand == 2:  # dilation
            im = cv2.dilate(im, kernel)
        # elif rand == 3:  # salt
        #     im = add_sorp(im, type="salt")
        # elif rand == 4:  # pepper
        #     im = add_sorp(im, type="pepper")
        return im

    def load_videos_track(self, shuffle=True):
        maxlen = self.args.batch_size  # get batch-size numbers frames
        vid_list = list(self.im_mani_root.iterdir())
        if shuffle:
            np.random.shuffle(vid_list)

        counter = 1
        Im = torch.empty(maxlen, 4, self.args.size, self.args.size)
        GT = torch.empty(maxlen, 1, self.args.size, self.args.size)

        for name in vid_list:
            prev_mask = None
            for i, _file in enumerate(name.iterdir()):
                if _file.suffix not in (".png", ".jpg"):
                    continue
                im_file = str(_file)
                mask_file = os.path.join(
                    str(self.mask_root), name.name, (_file.stem + ".png")
                )
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue

                image, mask = self.__get_im(im_file, mask_file, do_transform=False)

                if self.transform:
                    image_t, mask_t = self.transform(image, mask)

                if i == 0:
                    prev_mask = torch.zeros((1, self.args.size, self.args.size),
                                            dtype=torch.float32)

                Im[counter-1] = torch.cat((image_t, prev_mask), 0)
                GT[counter-1] = mask_t

                prev_mask = self.randomize_mask(mask)
                _, prev_mask = self.transform(None, prev_mask)

                if counter % maxlen == 0:
                    yield Im, GT
                    counter = 0
                counter += 1
        if counter < maxlen and counter > 1:
            yield Im, GT

    def load_data(self, batch=20, is_training=True, shuffle=True):
        counter = 0
        X = torch.empty((batch, 3, self.args.size, self.args.size), dtype=torch.float32)
        Y = torch.empty((batch, 1, self.args.size, self.args.size), dtype=torch.float32)
        Info = []

        if shuffle:
            np.random.shuffle(self.__im_files_with_gt)

        for i, im_file, mask_file in self.__im_files_with_gt:
            if (is_training and i in self.train_index) or (
                not is_training and i in self.test_index
            ):
                image, mask = self.__get_im(im_file, mask_file)
                X[counter] = image
                Y[counter] = mask
                Info.append((im_file, mask_file))
                counter += 1
                if counter % batch == 0:
                    if torch.any(torch.isnan(Y)):
                        import pdb; pdb.set_trace()
                    yield X, Y, Info
                    X = torch.empty(
                        (batch, 3, self.args.size, self.args.size), dtype=torch.float32
                    )
                    Y = torch.empty(
                        (batch, 1, self.args.size, self.args.size), dtype=torch.float32
                    )
                    Info = []
                    counter = 0
        if counter != 0:
            yield X, Y, Info

    # def __len__(self):
    #     return len(self.__im_files_with_gt)

    def __get_im(self, im_file, mask_file, do_transform=True):
        image = io.imread(im_file)
        image = skimage.img_as_float32(image)  # image in [0-1] range

        _mask = skimage.img_as_float32(io.imread(mask_file))

        if len(_mask.shape) > 2:
            ind = _mask[:, :, 2] > 0.5

            mask = np.zeros(_mask.shape[:2], dtype=np.float32)
            mask[ind] = 1
        else:
            mask = _mask

        if do_transform and self.transform is not None:
            image, mask = self.transform(image, mask)

        return image, mask

    def load_triplet(self, num=10):
        # randomly select one video and get frames (with labels)
        while True:
            name = np.random.choice(list(self.im_mani_root.iterdir()))
            gt_file = os.path.join(str(self.gt_root), name.name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = list(data.keys())

            list_forged_ind = []  # filename indices having forged part
            for i, f in enumerate(filenames):
                if data[f]["mask_orig"] is not None:
                    list_forged_ind.append(i)
            if list_forged_ind:
                break

        X_im = torch.empty(
            num, 3, 3, self.args.size, self.args.size, dtype=torch.float32
        )
        X_ind = torch.empty(num, 2, dtype=torch.float32)

        for i in range(num):
            ind = np.random.choice(list_forged_ind)
            cur_file = filenames[ind]
            cur_data = data[cur_file]

            mask_orig = cur_data["mask_orig"]
            mask_new = cur_data["mask_new"]
            offset = cur_data["offset"]
            src_file = filenames[ind - offset]

            # generate triplet: cur, im1 (with good match), im2 (random)
            src_neg_ind = np.random.choice(
                list(range(ind - offset))
                + list(range(ind - offset + 1, len(filenames)))
            )
            src_neg_file = filenames[src_neg_ind]

            #
            fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])
            im = skimage.img_as_float32(io.imread(fname))
            im_mask_new = mask_new

            im_t = self.image_with_mask(im, im_mask_new, type="foreground")
            # im_t = add_sorp(im_t, type="salt")

            src_fname = os.path.join(self.im_mani_root, *src_file.parts[-2:])
            im_src_pos = skimage.img_as_float32(io.imread(src_fname))
            ind_src_pos = (ind - offset) / len(filenames)

            if data[src_file]["mask_new"] is not None:
                _mask = data[src_file]["mask_new"]
                im_src_pos = self.image_with_mask(im_src_pos, _mask, type="background")
                # im_src_pos = add_sorp(im_src_pos, type="pepper")

            neg_fname = os.path.join(self.im_mani_root, *src_neg_file.parts[-2:])
            im_src_neg = skimage.img_as_float32(io.imread(neg_fname))
            ind_src_neg = src_neg_ind / len(filenames)

            if data[src_neg_file]["mask_new"] is not None:
                _mask = data[src_neg_file]["mask_new"]
                im_src_neg = self.image_with_mask(im_src_neg, _mask, type="background")
                # im_src_neg = add_sorp(im_src_neg, type="pepper")

            if self.transform:
                im_t = self.transform(im_t)
                im_src_pos = self.transform(im_src_pos)
                im_src_neg = self.transform(im_src_neg)

            X_im[i, 0] = im_t
            X_im[i, 1] = im_src_pos
            X_im[i, 2] = im_src_neg

            X_ind[i] = torch.tensor([ind_src_pos, ind_src_neg], dtype=torch.float32)
        return X_im, X_ind

    def image_with_mask(self, im, mask, type="foreground"):
        im = skimage.img_as_float32(im)
        mask = skimage.img_as_float32(mask)

        if len(im.shape) > len(mask.shape):
            mask = mask[..., None]

        if type == "foreground":
            im_masked = im * mask
        else:
            im_masked = im * (1 - mask)
        return im_masked

    def get_search_from_video(self):
        # randomly select one video and get frames (with labels)
        while True:
            name = np.random.choice(list(self.im_mani_root.iterdir()))
            gt_file = os.path.join(str(self.gt_root), name.name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = list(data.keys())
            flag = False
            for i, f in enumerate(filenames):
                if data[f]["mask_orig"] is not None:
                    flag = True
                    break
            if flag:
                break

        num = len(filenames)
        X_im = torch.empty(num, 3, self.args.size, self.args.size, dtype=torch.float32)

        _first = False
        for i in range(num):
            ind = i
            cur_file = filenames[ind]
            cur_data = data[cur_file]
            im_mask_new = cur_data["mask_new"]

            fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])
            im = skimage.img_as_float32(io.imread(fname))

            if im_mask_new is not None:
                if not _first:
                    im_first = self.image_with_mask(im, im_mask_new, type="foreground")
                    _first = True
                    first_ind = i
                    match_ind = i - cur_data["offset"]
                    im_first = self.transform(im_first)

                im = self.image_with_mask(im, im_mask_new, type="background")

            # im = add_sorp(im, type="pepper")

            if self.transform:
                im = self.transform(im)
            X_im[i] = im

        return X_im, im_first, match_ind, first_ind

    def get_frames_from_video(self, do_transform=False):
        # randomly select one video and get frames (with labels)
        name = np.random.choice(list(self.im_mani_root.iterdir()))
        for _file in name.iterdir():
            if _file.suffix == ".png":
                im_file = str(_file)
                mask_file = os.path.join(
                    str(self.mask_root), name.name, (_file.stem + ".png")
                )
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue
            image, mask = self.__get_im(im_file, mask_file)

            if torch.any(torch.isnan(mask)):
                import pdb; pdb.set_trace()

            yield image, mask

