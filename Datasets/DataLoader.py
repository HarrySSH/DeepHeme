import albumentations

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from torch.utils import data
## Simple augumentation to improtve the data generalibility




class Img_DataLoader(data.Dataset):
    def __init__(self, img_list='', in_dim=3, split='train', transform=False, in_size=96, df=None, encoder=None,
                 if_external=False):
        super(Img_DataLoader, self).__init__()
        self.split = split
        self.in_dim = in_dim
        self.transform = transform
        self.filelist = img_list
        self.in_size = in_size
        self.file_paths = img_list
        self.transform = transform
        self.df = df
        self.encoder = encoder
        self.if_external = if_external

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        sample = dict()
        img_path = self.file_paths[index]
        # prepare image
        orig_img = cv2.imread(img_path)
        image = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)

        '''
        if self.if_external:
            image = apply_brightness_contrast(image,-100,0)
        '''
        if self.transform is not None:
            try:
                img = self.transform(image=image)["image"]
            except:
                assert 1 == 2, 'something wrong'
                print(image)

        label = img_path.split('/')[-2]
        # print(img.shape)
        if self.if_external:
            img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_AREA)
        # img = img.reshape(3,96,96)

        img = np.einsum('ijk->kij', img)

        high_level_name = label
        if self.split != "compute":  # Use compute if you only want the prediction results. if you do this, make sure you don't shuffle the data
            mask = self.df[self.df['Cell_Types'] == high_level_name].iloc[:, 2:].to_numpy()
            length = mask.shape[1]
            sample["label"] = torch.from_numpy(mask.reshape(1, length)).float()  # one hot encoder

        sample["image"] = torch.from_numpy(img).float()  # self.encoder(torch.from_numpy(img).float())
        sample["ID"] = img_path
        return sample