"""
公共I/O工具函数

提供统一的医学影像数据读写、文件匹配功能，避免在各模型文件中重复定义。
"""

import os
import glob
import numpy as np
import SimpleITK as sitk
import nibabel as nib


def load_image(path):
    """
    读取CT图像（支持NIfTI和DICOM序列）

    Args:
        path: NIfTI文件路径或DICOM序列目录

    Returns:
        arr: numpy数组 (Z, Y, X)
        img: SimpleITK图像对象（保留元信息）
    """
    if os.path.isdir(path):
        reader = sitk.ImageSeriesReader()
        series_file_names = reader.GetGDCMSeriesFileNames(path)
        reader.SetFileNames(series_file_names)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # z, y, x
    return arr, img


def load_nifti(path):
    """
    读取 NIfTI 文件（返回完整元信息）

    Args:
        path: .nii.gz 文件路径

    Returns:
        img: SimpleITK图像对象
        arr: numpy数组 (Z, Y, X)
        spacing: 体素间距 (sx, sy, sz) mm
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    return img, arr, spacing


def load_seg(path):
    """
    读取分割掩膜，二值化

    Args:
        path: 掩膜文件路径

    Returns:
        arr: numpy数组 (Z, Y, X)，值为0或1
        img: SimpleITK图像对象
    """
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    arr = (arr > 0).astype(np.uint8)
    return arr, img


def save_nifti(array, reference_image, output_path):
    """
    保存为NIfTI文件，保留参考图像的空间信息

    Args:
        array: 要保存的numpy数组
        reference_image: 参考SimpleITK图像（用于复制空间信息）
        output_path: 输出路径
    """
    output_image = sitk.GetImageFromArray(array.astype(np.int16))
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, output_path)
    print(f"  Saved: {output_path}")


def save_nifti_from_affine(data, affine, path):
    """
    使用affine矩阵保存NIfTI文件（nibabel方式）

    Args:
        data: numpy数组
        affine: 4x4仿射矩阵
        path: 输出路径
    """
    nib.save(nib.Nifti1Image(data.astype(np.uint8), affine), path)


def match_image_mask_pairs(image_dir, mask_dir,
                           image_suffix=".nii.gz",
                           mask_suffix=".nii.gz"):
    """
    从两个文件夹中按文件名匹配 image 和 mask

    Args:
        image_dir: 图像文件夹路径
        mask_dir: 掩膜文件夹路径
        image_suffix: 图像文件后缀
        mask_suffix: 掩膜文件后缀

    Returns:
        list: [(case_id, image_path, mask_path), ...]
    """
    image_files = sorted(glob.glob(os.path.join(image_dir, f"*{image_suffix}")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, f"*{mask_suffix}")))

    def extract_case_id(path, suffix):
        name = os.path.basename(path)
        if name.endswith(suffix):
            name = name[:-len(suffix)]
        for tag in ["_ct", "_resample", "_mask", "_image", "_seg", "_label", "_tumor"]:
            name = name.replace(tag, "")
        return name.strip("_")

    image_dict = {extract_case_id(f, image_suffix): f for f in image_files}
    mask_dict = {extract_case_id(f, mask_suffix): f for f in mask_files}

    pairs = []
    for case_id in sorted(set(image_dict.keys()) & set(mask_dict.keys())):
        pairs.append((case_id, image_dict[case_id], mask_dict[case_id]))

    return pairs
