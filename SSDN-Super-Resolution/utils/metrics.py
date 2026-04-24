import numpy as np
import math
import cv2


def rgb2y(img):
    y = np.dot(img, [0.256789, 0.504129, 0.097906]) + 16.0
    return y


def calculate_psnr(img1, img2, crop_border=0, test_y_channel=True):

    assert img1.shape == img2.shape, "Images must have the same shape"
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # 1. 边界裁剪
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    # 2. RGB 转 Y 通道
    if test_y_channel and img1.ndim == 3 and img1.shape[2] == 3:
        img1 = rgb2y(img1)
        img2 = rgb2y(img2)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def _ssim(img1, img2):

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq, mu2_sq = mu1 ** 2, mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2, crop_border=0, test_y_channel=True):

    assert img1.shape == img2.shape, "Images must have the same shape"
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel and img1.ndim == 3 and img1.shape[2] == 3:
        img1 = rgb2y(img1)
        img2 = rgb2y(img2)
        return _ssim(img1, img2)
    elif img1.ndim == 2 or (img1.ndim == 3 and img1.shape[2] == 1):
        return _ssim(img1.squeeze(), img2.squeeze())
    else:
        ssims = [_ssim(img1[..., i], img2[..., i]) for i in range(3)]
        return np.mean(ssims)