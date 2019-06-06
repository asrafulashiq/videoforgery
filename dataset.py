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

def get_boundary(im):
    kernel = np.ones((5, 5), dtype=np.float32)
    im_bnd = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, kernel)
    return im_bnd


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
                    if not os.path.exists(mask_file):
                        mask_file = os.path.join(
                            str(self.mask_root), name.name, (_file.stem + ".jpg")
                        )
                    try:
                        assert os.path.exists(mask_file)
                    except AssertionError:
                        raise FileNotFoundError(f"{mask_file} not found")
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


    def load_videos_all(self):
        for ind in self.test_index:
            D = self.data[ind]
            name = D['name']
            files = D['files']
            gt_file = os.path.join(str(self.gt_root), Path(name).name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = list(data.keys())
            offset = data[filenames[0]]["offset"]

            _len = len(filenames)
            X = np.zeros((_len, self.args.size, self.args.size, 3), dtype=np.float32)
            Y_red = np.zeros((_len, self.args.size, self.args.size), dtype=np.float32)
            Y_green = np.zeros((_len, self.args.size, self.args.size), dtype=np.float32)

            flag = False
            forge_time = None
            gt_time = None
            for i, cur_file in enumerate(sorted(filenames)):
                cur_data = data[cur_file]
                mask_orig = cur_data["mask_orig"]
                mask_new = cur_data["mask_new"]
                offset = cur_data["offset"]

                if mask_orig is not None and not flag:
                    forge_time = [i, -1]
                    gt_time = [i-offset, -1]
                    flag = True

                if mask_orig is None and flag:
                    gt_time[1] = i
                    forge_time[1] = i - offset

                fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])
                im = skimage.img_as_float32(io.imread(fname))

                X[i] = cv2.resize(im, (self.args.size, self.args.size))
                if mask_new is None:
                    mask_new = np.zeros((self.args.size, self.args.size), dtype=np.float32)
                    mask_orig = np.zeros((self.args.size, self.args.size), dtype=np.float32)
                Y_red[i] = cv2.resize(mask_orig, (self.args.size, self.args.size))
                Y_green[i-offset] = cv2.resize(mask_new, (self.args.size, self.args.size))

            if forge_time is not None and forge_time[1] == -1:
                forge_time[1] = i
                gt_time[1] = i - offset
            yield X, Y_red, forge_time, Y_green, gt_time, name


    def load_videos_track(self, is_training=True, add_prev=True, is_shuffle=True):

        idx = []
        if is_training:
            idx = self.train_index
        else:
            idx = self.test_index

        if is_shuffle:
            np.random.shuffle(idx)

        for cnt, _ind in enumerate(idx):
            inf = self.data[_ind]
            if is_training:
                maxlen = self.args.batch_size
            else:
                maxlen = len(inf["files"])

            if add_prev:
                dim = 4
            else:
                dim=3

            Im = torch.zeros(maxlen, dim, self.args.size, self.args.size)
            GT = torch.zeros(maxlen, 1, self.args.size, self.args.size)

            prev_mask = None
            counter = 1

            for i, (im_file, mask_file) in enumerate(inf["files"]):
                im_file = str(im_file)
                mask_file = str(mask_file)
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue

                image, mask = self.__get_im(im_file, mask_file, do_transform=False)

                if self.transform:
                    image_t, mask_t = self.transform(image, mask)

                if add_prev and i == 0:
                    prev_mask = torch.zeros(
                        (1, self.args.size, self.args.size), dtype=torch.float32
                    )

                if add_prev:
                    Im[counter - 1] = torch.cat((image_t, prev_mask), 0)
                else:
                    Im[counter - 1] = image_t
                GT[counter - 1] = mask_t

                if add_prev:
                    prev_mask = self.randomize_mask(mask)
                    _, prev_mask = self.transform(None, prev_mask)

                if counter % maxlen == 0:
                    yield Im, GT
                    Im = torch.zeros(maxlen, dim, self.args.size, self.args.size)
                    GT = torch.zeros(maxlen, 1, self.args.size, self.args.size)
                    counter = 0
                counter += 1

    def load_data(self, batch=20, is_training=True, shuffle=True, with_boundary=False):
        counter = 0
        X = torch.zeros((batch, 3, self.args.size, self.args.size), dtype=torch.float32)
        if with_boundary:
            ysize = 2
        else:
            ysize = 1
        Y = torch.zeros((batch, ysize, self.args.size, self.args.size), dtype=torch.float32)

        Info = []

        if shuffle:
            np.random.shuffle(self.__im_files_with_gt)

        for i, im_file, mask_file in self.__im_files_with_gt:
            if (is_training and i in self.train_index) or (
                not is_training and i in self.test_index
            ):
                tmp = self.__get_im(im_file, mask_file, with_boundary=with_boundary)
                X[counter] = tmp[0]
                Y[counter, 0] = tmp[1]
                if with_boundary:
                    Y[counter, 1] = tmp[2]
                Info.append((im_file, mask_file))
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
                        (batch, ysize, self.args.size, self.args.size), dtype=torch.float32
                    )
                    Info = []
                    counter = 0

    # def __len__(self):
    #     return len(self.__im_files_with_gt)

    def __get_im(self, im_file, mask_file, do_transform=True, with_boundary=False):
        image = io.imread(im_file)
        image = skimage.img_as_float32(image)  # image in [0-1] range

        _mask = skimage.img_as_float32(io.imread(mask_file))

        if len(_mask.shape) > 2:
            ind = _mask[:, :, 2] > 0.5

            mask = np.zeros(_mask.shape[:2], dtype=np.float32)
            mask[ind] = 1
        else:
            mask = _mask

        if with_boundary:
            boundary = get_boundary(mask)

        if do_transform and self.transform is not None:
            image, mask = self.transform(image, mask)

            if torch.any(mask < 0):
                import pdb

                pdb.set_trace()

            if with_boundary:
                _, boundary = self.transform(None, boundary)
        if with_boundary:
            return image, mask, boundary
        else:
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
        return im_masked

    def get_search_from_video(self, first_only=True):
        # randomly select one video and get frames (with labels)
        lists = list(self.im_mani_root.iterdir())
        np.random.shuffle(lists)
        for name in lists:
            # while True:
            #     name = np.random.choice(list(self.im_mani_root.iterdir()))
            gt_file = os.path.join(str(self.gt_root), name.name + ".pkl")

            with open(gt_file, "rb") as fp:
                data = pickle.load(fp)

            filenames = list(data.keys())
            flag = False
            for i, f in enumerate(filenames):
                if data[f]["mask_orig"] is not None:
                    flag = True
                    break
            if not flag:
                continue

            num = len(filenames)
            # X_im = torch.empty(num, 3, self.args.size, self.args.size, dtype=torch.float32)
            X_im = np.zeros((num, self.args.size, self.args.size, 3), dtype=np.float32)

            _first = False
            X_ref = None
            for i in range(num):
                ind = i
                cur_file = filenames[ind]
                cur_data = data[cur_file]
                im_mask_new = cur_data["mask_new"]

                fname = os.path.join(self.im_mani_root, *cur_file.parts[-2:])
                im = skimage.img_as_float32(io.imread(fname))

                if im_mask_new is not None:
                    if not _first:
                        im_first = self.image_with_mask(
                            im, im_mask_new, type="foreground-bbox"
                        )
                        _first = True
                        first_ind = i
                        match_ind = i - cur_data["offset"]
                        im_first = cv2.resize(
                            im_first, (self.args.size, self.args.size)
                        )

                        # im_first = self.transform(im_first)
                    if not first_only:
                        if X_ref is None:
                            X_ref = im_first[None, ...]
                        else:
                            im_ref = self.image_with_mask(
                                im, im_mask_new, type="foreground-bbox"
                            )
                            im_ref = cv2.resize(
                                im_ref, (self.args.size, self.args.size)
                            )
                            X_ref = np.concatenate((X_ref, im_ref[None, ...]), 0)

                    im = self.image_with_mask(im, im_mask_new, type="background-bbox")

                # if self.transform:
                #     im = self.transform(im)
                im = cv2.resize(im, (self.args.size, self.args.size))
                X_im[i] = im
            yield X_im, X_ref, match_ind, first_ind

    def get_frames_from_video(self, do_transform=False,
                              get_all=False, is_test=False):
        # randomly select one video and get frames (with labels)
        name = np.random.choice(self.data)
        for im_file, mask_file in name["files"]:
            if Path(im_file).suffix == ".png":
                im_file = str(im_file)
                mask_file = str(mask_file)
                try:
                    assert os.path.exists(mask_file)
                except AssertionError:
                    continue
            image, mask = self.__get_im(im_file, mask_file, do_transform=do_transform)

            yield image, mask
