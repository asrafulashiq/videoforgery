import os
from pathlib import Path
from matplotlib import cm

import matplotlib.pyplot as plt
import numpy as np

# import seaborn as sns
import argparse
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import pickle
import cv2
from PIL import Image


def set_axes_radius(ax, origin, radius):
    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def set_axes_equal(ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    set_axes_radius(ax, origin, radius)


Datasets=["davis_tempered", "SegTrackv2_tempered",
        "youtube_tempered"]

if __name__ == "__main__":


    for each in Datasets:
        root = Path("./data/") / each / "gt"

        scale = 1 / 10

        color1 = (128, 0, 0)
        color2 = (0, 128, 0)

        for each_pkl in tqdm(root.iterdir()):
            with each_pkl.open("rb") as fp:
                Data = pickle.load(fp)

            T = len(Data)

            for i, fname in tqdm(enumerate(Data)):
                mask_orig = Data[fname]["mask_orig"]
                mask_new = Data[fname]["mask_new"]
                offset = Data[fname]["offset"]

                empty = False
                if mask_orig is None:
                    empty = True  # no mask here

                if i == 0:
                    # r, c = int(mask_new.shape[0]*scale), int(mask_new.shape[1]*scale)
                    r, c = 20, 20
                    X1 = np.zeros((T, r, c), dtype=bool)
                    X2 = np.zeros((T, r, c), dtype=bool)

                if empty:
                    mask_new = np.zeros((r, c), dtype=np.uint8)
                    mask_orig = np.zeros((r, c), dtype=np.uint8)
                else:
                    mask_new = cv2.resize(mask_new, (c, r), interpolation=cv2.INTER_NEAREST)
                    mask_orig = cv2.resize(mask_orig, (c, r), interpolation=cv2.INTER_NEAREST)

                # mask_orig = cv2.resize(mask_orig, (c, r), interpolation=cv2.INTER_NEAREST)
                if not empty:
                    X1[i - offset] = mask_orig
                    X2[i] = mask_new > 0



            ind = np.random.choice(T, size=min(T, 60), replace=False)
            ind.sort()

            X1 = X1[ind]
            X2 = X2[ind]

            X1 = X1.transpose(2, 0, 1)
            X2 = X2.transpose(2, 0, 1)

            Colors1 = np.zeros(X1.shape + (4,), dtype=np.float)
            Colors2 = np.zeros(X2.shape + (4,), dtype=np.float)

            # Colors1[:, :] = (0, 0, 0, 0)
            # Colors2[:, :] = (0, 0, 0, 0)

            Colors1[X1] = (1, 0, 0, 0.8)
            Colors2[X2] = (0, 0, 1, 0.5)

            print(X1.shape)

            # sns.set_style("white")
            scale_x, scale_y, scale_z = (0.6, 1.7, 0.6)

            fig = plt.figure(figsize=(8,6))
            ax = fig.gca(projection="3d")

            ax.get_proj = lambda: np.dot(
                Axes3D.get_proj(ax), np.diag([scale_x, scale_y, scale_z, 1])
            )

            ax.voxels(X1, facecolors=Colors1, label="source")
            ax.voxels(X2, facecolors=Colors2, label="manipulated")
            # ax.legend()

            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])
            # ax.set_xlabel("x")
            ax.view_init(elev=66, azim=-40)
            # plt.show()
            # _dir = "./data/davis_tempered/fig"
            _dir = '/'.join(root.parts[:-1]) + '/fig'
            if not os.path.exists(_dir):
                os.mkdir(_dir)

            plt.savefig("{}/{}.png".format(_dir, each_pkl.stem))
            plt.close("all")
            # break