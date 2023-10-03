# importing the libraries

from __future__ import print_function

import argparse

import os
import sys
import pandas as pd
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import cv2
import  glob
import time
import albumentations
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder# creating instance of one-hot-encoder
### Internal Imports
from models.ResNext50 import Myresnext50
from train.train_classification import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import Img_DataLoader

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import glob

def main(args):

    meta_data = pd.read_csv(args.meta_data)
    X_train = meta_data[meta_data['split']=='train']['fpath'].tolist()
    X_val = meta_data[meta_data['split']=='val']['fpath'].tolist()

    labels = [x.split('/')[-2] for x in X_train]
    cell_types = set(labels)

    cell_types = list(cell_types)
    cell_types.sort()

    cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])# converting type of columns to 'category'
    cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')# Assigning numerical values and storing in another column
    cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes



    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())
    cell_types_df = cell_types_df.join(enc_df)

    # load model

    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  # Interesting! This worked for no reason haha
    if args.input_model == 'ResNeXt50':
        resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=args.pretrained)
        my_extended_model = Myresnext50(my_pretrained_model= resnext50_pretrained, num_classes = len(cell_types))

    ## Simple augumentation to improtve the data generalibility

    val_transform_pipeline = albumentations.Compose(
        [
            albumentations.Normalize(mean=(0.5642, 0.5026, 0.6960), std=(0.2724,
 0.2838, 0.2167)),

        ]
    )

    train_transform_pipeline = albumentations.Compose([
    albumentations.ShiftScaleRotate(p = 0.8),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.VerticalFlip(p=0.5),
    albumentations.Affine(shear=(-10,10), p = 0.3),
    albumentations.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.01), always_apply=False, p=0.2),
    albumentations.RandomBrightnessContrast(contrast_limit=0.4, brightness_by_max=0.4, p=0.5),
    albumentations.CLAHE(p = 0.3),
    albumentations.ColorJitter(p = 0.2),
    albumentations.RandomGamma(p = 0.2),
])

    trainer = trainer_classification(train_image_files=X_train, validation_image_files=X_val, model=my_extended_model,
                                        train_img_transform=train_transform_pipeline,
                                     val_img_transform=val_transform_pipeline, init_lr=args.init_lr,
                                     lr_decay_every_x_epochs=args.lr_decay_every_x_epochs,

                                     weight_decay=args.weight_decay, batch_size=args.batch_size, epochs=args.epochs, gamma=args.gamma, df=cell_types_df,
                                     save_checkpoints_dir=args.save_checkpoints_dir)

    My_model = trainer.train(my_extended_model)


# Training settings
parser = argparse.ArgumentParser(description='Configurations for Model training')
parser.add_argument('--meta_data', type=str,
                    default='/data/aa-ssun2-cmp/hemepath_dataset_FINAL/metadata/data_info.csv',)
                    

parser.add_argument('--input_model', type=str,
                    default='ResNeXt50',
                    help='input model, the defulat is the pretrained model')

parser.add_argument('--pretrained', type=bool,
                    default=True,
                    help='the defulat is the pretrained model')

parser.add_argument('--init_lr', type=float,
                    default=0.001,
                    help='learning rate')

parser.add_argument('--weight_decay', type=float,
                    default=0.0005,
                    help='weight decay')

parser.add_argument('--gamma', type=float,
                    default=0.1,
                    help='gamma')

parser.add_argument('--epochs', type=float,
                    default=100,
                    help='epoch number')

parser.add_argument('--batch_size', type=int,
                    default=1024,
                    help='epoch number')

parser.add_argument('--lr_decay_every_x_epochs', type = int,
                    default=10,
                    help='learning rate decay per X step')

parser.add_argument('--save_checkpoints_dir', type = str,
                    default=None,
                    help='save dir')

args = parser.parse_args()

if __name__ == "__main__":
    main(args)
    print('Done')