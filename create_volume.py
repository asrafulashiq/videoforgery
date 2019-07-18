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
import skimage

import matplotlib.pyplot as plt


def plot_conf_mat(conf_arr, path):
    norm_conf = []
    for i in conf_arr:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a + 1e-8))
        norm_conf.append(tmp_arr)
    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    interpolation='nearest')
    width, height = np.array(conf_arr).shape
    for x in range(width):
        for y in range(height):
            ax.annotate(f"{conf_arr[x][y]:.2f}", xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center')

    # cb = fig.colorbar(res)
    classnames = [str(i) for i in range(width)]
    plt.xticks(range(width), classnames[:width])
    plt.yticks(range(height), classnames[:height])
    plt.savefig(path, format='png')
    plt.close('all')


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

def create(Ds, Df, path=None, size=(30, 30), t_len=None):
    T = Ds.shape[0]

    r, c = size

    X1 = skimage.transform.resize(Ds, (Ds.shape[0], r, c), order=0)
    X2 = skimage.transform.resize(Df, (Df.shape[0], r, c), order=0)

    X1 = X1 > 0.5
    X2 = X2 > 0.5
    
    if t_len is not None:
        ind = np.random.choice(T, size=min(T, t_len), replace=False)
        ind.sort()
        X1 = X1[ind]
        X2 = X2[ind]

    X1 = X1.transpose(2, 0, 1)
    X2 = X2.transpose(2, 0, 1)

    Colors1 = np.zeros(X1.shape + (4,), dtype=np.float)
    Colors2 = np.zeros(X2.shape + (4,), dtype=np.float)

    Colors1[X1] = (0, 1., 0, 0.5)
    Colors2[X2] = (1., 0, 0, 0.5)

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
    ax.set_yticks([])  # time axis
    ax.set_zticks([])

    ax.view_init(elev=24, azim=-52)
    # plt.show()
    # _dir = "./data/davis_tempered/fig"
    if path is not None:
        fig.savefig(str(path))
        plt.close("all")
