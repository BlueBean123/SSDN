import os
import random
from typing import Tuple
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

IMG_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    return filename.lower().endswith(IMG_EXTENSIONS)


class SRDataset(Dataset):
    """
    Dataset loader for Image Super-Resolution.

    Args:
        lr_path (str): Directory containing Low-Resolution images.
        hr_path (str): Directory containing High-Resolution images.
        patch_size (int): Spatial size of the cropped LR patches.
        scale (int): Super-resolution upsampling scale factor.
        is_train (bool): If True, applies random crop and augmentations.
    """

    def __init__(self, lr_path: str, hr_path: str, patch_size: int = 48, scale: int = 2, is_train: bool = True):
        super(SRDataset, self).__init__()
        self.scale = scale
        self.patch_size = patch_size
        self.is_train = is_train

        self.hr_filenames = sorted([os.path.join(hr_path, x) for x in os.listdir(hr_path) if is_image_file(x)])
        self.lr_filenames = sorted([os.path.join(lr_path, x) for x in os.listdir(lr_path) if is_image_file(x)])

        if len(self.hr_filenames) != len(self.lr_filenames):
            raise ValueError(f"Dataset size mismatch: HR ({len(self.hr_filenames)}) vs LR ({len(self.lr_filenames)}).")

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        lr = Image.open(self.lr_filenames[index]).convert('RGB')
        hr = Image.open(self.hr_filenames[index]).convert('RGB')

        if self.is_train:
            hr_patch_size = self.patch_size * self.scale
            w_lr, h_lr = lr.size

            # Random crop based on LR spatial dimensions
            if w_lr > self.patch_size and h_lr > self.patch_size:
                x = random.randint(0, w_lr - self.patch_size)
                y = random.randint(0, h_lr - self.patch_size)

                lr = lr.crop((x, y, x + self.patch_size, y + self.patch_size))
                hr = hr.crop((x * self.scale, y * self.scale,
                              x * self.scale + hr_patch_size, y * self.scale + hr_patch_size))

            # Random geometric augmentations
            aug_mode = random.randint(0, 3)
            if aug_mode == 1:
                lr, hr = TF.hflip(lr), TF.hflip(hr)
            elif aug_mode == 2:
                lr, hr = TF.vflip(lr), TF.vflip(hr)
            elif aug_mode == 3:
                lr, hr = TF.rotate(lr, 90), TF.rotate(hr, 90)

        lr_tensor = TF.to_tensor(lr)
        hr_tensor = TF.to_tensor(hr)

        return lr_tensor, hr_tensor

    def __len__(self) -> int:
        return len(self.hr_filenames)