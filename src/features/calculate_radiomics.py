# 对分块提取影像组学特征
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor 
# 特征计算函数
def catch_features(imagePath, maskPath, i):
    settings = {}
    settings['binWidth'] = 25  
    settings['label'] = i
    settings['Interpolator'] = 3
    settings['resampledPixelSpacing'] = [1, 1, 1]
    settings['normalize'] = True
    settings['normalizeScale'] = 255
    # settings['sigma'] = [1,2,3,4,5]
    # settings['wavelet'] = 'db'

    # 提取特征
    extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    # print('Extraction parameters:\n\t', extractor.settings)
    # extractor.enableImageTypeByName('LoG')
    # extractor.enableImageTypeByName('Wavelet')
    extractor.enableAllFeatures()

    result = extractor.execute(imagePath, maskPath)
    return result

data_path = './data/TongJi/'
rad_df = pd.DataFrame()
image_path = os.path.join(data_path, 'image_resample')
mask_path = os.path.join(data_path, 'mask_resample')
for name in os.listdir(image_path):
    image_name = os.path.join(image_path, name)
    mask_name = os.path.join(mask_path, name.replace("image.nii", "seg.nii"))
    
    result = catch_features(image_name, mask_name, 1)

    rad_series = pd.DataFrame(result.values(), index=result.keys(), columns=[name.replace("image.nii.gz", "")])

    rad_df = pd.concat([rad_df, rad_series], axis=1)
    print(rad_df)
rad_df = rad_df.iloc[38:, :].T
# rad_df.to_csv(os.path.join(data_path, 'rad_features.csv'))
rad_df.to_excel(os.path.join(data_path, 'rad_features.xlsx'))

