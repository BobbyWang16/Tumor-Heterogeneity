import numpy as np
import os
import SimpleITK as sitk
for dataset in os.listdir("./data"):
    data_path = f"./data/{dataset}/image_process"
    save_path = f"./data/{dataset}/seg"
    if not os.path.exists(data_path):
        continue
    
    for file in os.listdir("./pred_a"):
        a = sitk.ReadImage("./pred_a/"+file)
        v = sitk.ReadImage("./pred_v/"+file)
        a_arr = sitk.GetArrayFromImage(a)
        v_arr = sitk.GetArrayFromImage(v)

        new_arr = np.zeros_like(a_arr)
        new_arr[a_arr > 0] = 1
        new_arr[v_arr > 0] = 2
        new_arr = new_arr.astype(np.uint8)
        new_image = sitk.GetImageFromArray(new_arr)
        new_image.CopyInformation(a)
        sitk.WriteImage(new_image, "./"+file)