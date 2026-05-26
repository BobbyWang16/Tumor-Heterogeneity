"""
公共图像预处理工具函数

提供肿瘤定位、HU裁剪、连通域处理、重采样等通用预处理操作。
"""

import numpy as np
from scipy import ndimage as ndi


def keep_largest_cc(mask):
    """
    保留最大连通域（去除噪声/小区域）

    Args:
        mask: 二值掩膜数组

    Returns:
        只保留最大连通域的二值掩膜
    """
    lab, n = ndi.label(mask.astype(np.uint8))
    if n <= 1:
        return mask.astype(bool)
    counts = np.bincount(lab.ravel())
    counts[0] = 0
    keep = counts.argmax()
    return (lab == keep)


def locate_tumor_3d(vol, msk, padding=2):
    """
    根据掩膜定位肿瘤区域，裁剪出包含肿瘤的ROI

    Args:
        vol: 3D图像数组 (Z, Y, X)
        msk: 3D掩膜数组 (Z, Y, X)
        padding: 边界填充像素数

    Returns:
        sub_vol: 裁剪后的图像ROI
        sub_msk: 裁剪后的掩膜ROI
        offset: 裁剪起点坐标 (z0, y0, x0)
    """
    coords = np.argwhere(msk > 0)
    if coords.size == 0:
        return None, None, None
    z_min, y_min, x_min = coords.min(axis=0)
    z_max, y_max, x_max = coords.max(axis=0)
    z0 = max(z_min - padding, 0)
    z1 = min(z_max + padding + 1, vol.shape[0])
    y0 = max(y_min - padding, 0)
    y1 = min(y_max + padding + 1, vol.shape[1])
    x0 = max(x_min - padding, 0)
    x1 = min(x_max + padding + 1, vol.shape[2])

    sub_vol = vol[z0:z1, y0:y1, x0:x1]
    sub_msk = msk[z0:z1, y0:y1, x0:x1]
    return sub_vol, sub_msk, (z0, y0, x0)


def clip_hu(image_array, mask_array, hu_min, hu_max):
    """
    按HU值范围裁剪图像，掩膜外区域置为hu_min

    Args:
        image_array: 3D CT图像数组
        mask_array: 3D掩膜数组
        hu_min: HU下限
        hu_max: HU上限

    Returns:
        裁剪后的图像数组
    """
    clipped = np.clip(image_array, hu_min, hu_max)
    clipped[mask_array == 0] = hu_min
    return clipped


def crop_to_mask(img_np, mask_np, margin=2):
    """
    根据掩膜裁剪图像，保留margin像素边距

    Args:
        img_np: 3D图像数组
        mask_np: 3D掩膜数组
        margin: 边距像素数

    Returns:
        裁剪后的图像和掩膜
    """
    idx = np.where(mask_np)
    if len(idx[0]) == 0:
        return img_np, mask_np
    z_min, y_min, x_min = [i.min() for i in idx]
    z_max, y_max, x_max = [i.max() for i in idx]
    z0 = max(z_min - margin, 0)
    z1 = min(z_max + margin + 1, img_np.shape[0])
    y0 = max(y_min - margin, 0)
    y1 = min(y_max + margin + 1, img_np.shape[1])
    x0 = max(x_min - margin, 0)
    x1 = min(x_max + margin + 1, img_np.shape[2])
    return (img_np[z0:z1, y0:y1, x0:x1],
            mask_np[z0:z1, y0:y1, x0:x1])
