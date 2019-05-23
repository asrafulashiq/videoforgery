from torch import nn
from torch.nn import functional as F
import torch
from unet_models import UNet11
from torchvision import transforms
import numpy as np
import cv2

class CustomTransform():
    def __init__(self, size=224):
        self.size = (size, size)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        # self.normalize = transforms.Normalize(mean, std)
        self.to_tensor = transforms.ToTensor()

    def __call__(self, img, mask=None):
        if img.dtype == np.uint8:
            img = (img / 255.).astype(np.float32)

        img = cv2.resize(img, self.size)
        img = (img - self.mean) / self.std
        img = self.to_tensor(img)

        if mask is not None:
            mask = cv2.resize(mask, self.size)
            mask = self.to_tensor(mask)

        return img, mask