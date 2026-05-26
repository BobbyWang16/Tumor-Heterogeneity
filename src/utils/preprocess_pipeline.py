import os
import SimpleITK as sitk
import numpy as np
from skimage.measure import label, regionprops

# dataset_list = ['immunotherapy', 'NSCLC-Radiomics-Genomics', 'LungCT-Diagnosis', 'NSCLC_Radiogenomics', 'NSCLC-Radiomics', 'NSCLC-Radiomics-Interobserver1', 'TCGA', 'CPTAC']
dataset_list = ['TongJi']

for dataset in dataset_list:
    imagepath = f"./data/{dataset}/image"
    maskpath = f"./data/{dataset}/mask"
    mask_savepath = f"./data/{dataset}/mask_resample0"
    image_savepath = f"./data/{dataset}/image_resample0"

    if not os.path.exists(image_savepath):
        os.makedirs(image_savepath)
    if not os.path.exists(mask_savepath):
        os.makedirs(mask_savepath)

    for file in os.listdir(imagepath):
        if os.path.exists(os.path.join(f"./data/{dataset}/image_resample", file)):
            continue
        print(dataset, file)
        image = sitk.ReadImage(os.path.join(imagepath, file))
        mask = sitk.ReadImage(os.path.join(maskpath, file))
        print(image.GetDirection())

        mask_array = sitk.GetArrayFromImage(mask)
        mask_array = label(mask_array)
        regions = regionprops(mask_array)
        # assert len(regions) == 1, f"More than one region in mask {file}, {len(regions)}"

        if len(regions) > 1:
            region = max(regions, key=lambda r: r.area)
            mask_array = np.zeros_like(mask_array)
            mask_array[region.coords[:, 0], region.coords[:, 1], region.coords[:, 2]] = 1
            mask_array = mask_array.astype(np.uint8)
            new_mask = sitk.GetImageFromArray(mask_array)
            new_mask.CopyInformation(mask)

        # Resample the image and mask (1, 1, 1) mm spacing, image choose B-splin interpolation, mask choose nearest neighbor interpolation
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing((1.0, 1.0, 1.0))

        # calculating the new size
        new_size = [int(image.GetSize()[i] * image.GetSpacing()[i] / resample_filter.GetOutputSpacing()[i]) for i in range(3)]
        resample_filter.SetSize(new_size)
        # resample_filter.SetSize(image.GetSize())
        resample_filter.SetOutputOrigin(image.GetOrigin())
        resample_filter.SetOutputDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
        resample_filter.SetInterpolator(sitk.sitkBSpline)
        resample_filter.SetTransform(sitk.Transform())
        # resample_filter.SetDefaultPixelValue(image.GetPixelIDValue())
        # resample_filter.SetOutputPixelType(image.GetPixelIDValue())
        # resample_filter.SetInput(image)
        resample_image = resample_filter.Execute(image)
        sitk.WriteImage(resample_image, os.path.join(image_savepath, file))

        resample_filter.SetInterpolator(sitk.sitkNearestNeighbor)
        # resample_filter.SetInput(mask)
        resample_mask = resample_filter.Execute(new_mask)
        sitk.WriteImage(resample_mask, os.path.join(mask_savepath, file))