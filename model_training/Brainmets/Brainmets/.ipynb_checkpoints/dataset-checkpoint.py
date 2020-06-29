import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from imblearn.over_sampling import RandomOverSampler
import skimage
from Brainmets.utils import *

class MetDataSet(Dataset):
    def __init__(self, df, transformer = None, resample = None, clahe = None):
        if resample == 'max_size':
            ro = RandomOverSampler()
            df['max_size_class'] = df['max_met_size']>100
            X_resampled, y_resampled = ro.fit_resample(df.drop(columns=['max_size_class']),df['max_size_class'])
            df = pd.concat([X_resampled, y_resampled], axis = 1)
        self.img_files = list(df['img_files'])
        self.mask_files = list(df['mask_files'])
        self.img_names = ['_'.join(file.split('/')[-1].split('_')[0:2]) for file in self.img_files]
        self.mask_names = ['_'.join(file.split('/')[-1].split('_')[0:2]) for file in self.mask_files]
        self.transformer = transformer
        self.clahe = cv2.createCLAHE() if clahe else None
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img = read_and_crop(self.img_files[idx],64,256,256).reshape(1,64,256,256)
        mask = read_and_crop(self.mask_files[idx],64,256,256).reshape(1,64,256,256)
        if self.transformer:
            img, mask =self.transformer.transform(img[0], mask[0])
            img = img.reshape(1,64,256,256)
            mask = mask.reshape(1,64,256,256)
        if self.clahe:
            he_img = cv2.normalize(img[0], None, alpha = 0, beta = 255,
                                       norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
            he_img = he_img.astype(np.uint8)
            he_img = np.concatenate(he_img)
            he_img = self.clahe.apply(he_img)
            he_img = skimage.img_as_float(he_img).reshape(1,64,256,256)
            return he_img.copy(), mask.copy()
        return img.copy(), mask.copy()
    def get_name(self,idx):
        img_name = self.img_names[idx]
        mask_name = self.mask_names[idx]
        return img_name,mask_name