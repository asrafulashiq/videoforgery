from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11
from torchvision import transforms
import numpy as np


class CustomTransform():
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask=None):
        if img.dtype == np.uint8:
            img = img / 255.

        img = (img - self.mean) / self.std
        img = self.to_tensor(img)

        if mask is not None:
            mask = self.to_tensor(mask)

        return img, mask