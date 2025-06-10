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
import glob
import time
import albumentations
import numpy as np
from sklearn.preprocessing import OneHotEncoder

### Internal Imports
from models.CNN_models import CNNModel  # Changed from CNNModels
from train.train_classification import trainer_classification
from utils.utils import configure_optimizers
from Datasets.DataLoader import ImageDataset

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

def main(args):
    meta_data = pd.read_csv(args.meta_data)
    X_train = meta_data[meta_data['split']=='train']['fpath'].tolist()
    X_val = meta_data[meta_data['split']=='val']['fpath'].tolist()

    labels = [x.split('/')[-2] for x in X_train]
    cell_types = sorted(set(labels))

    cell_types_df = pd.DataFrame(cell_types, columns=['Cell_Types'])
    cell_types_df['Cell_Types'] = cell_types_df['Cell_Types'].astype('category')
    cell_types_df['Cell_Types_Cat'] = cell_types_df['Cell_Types'].cat.codes

    enc = OneHotEncoder(handle_unknown='ignore')
    enc_df = pd.DataFrame(enc.fit_transform(cell_types_df[['Cell_Types_Cat']]).toarray())
    cell_types_df = cell_types_df.join(enc_df)

    # Load model
    torch.hub._validate_not_a_forked_repo = lambda a, b, c: True  
    if args.input_model == 'ResNeXt50':
        resnext50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=resnext50_pretrained, num_classes=len(cell_types))
    elif args.input_model == "GoogleNet":
        googlenet_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=googlenet_pretrained, num_classes=len(cell_types))
    elif args.input_model == "Inception_v3":
        inception_v3_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=inception_v3_pretrained, num_classes=len(cell_types))
    elif args.input_model == "vgg19":
        vgg19_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=vgg19_pretrained, num_classes=len(cell_types))
    elif args.input_model == "efficientnet_v2":
        efficientnet_v2_pretrained = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_v2_rw_m', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=efficientnet_v2_pretrained, num_classes=len(cell_types))
    elif args.input_model == "resnet50":
        resnet50_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=resnet50_pretrained, num_classes=len(cell_types))
    elif args.input_model == "resnet101":
        resnet101_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=resnet101_pretrained, num_classes=len(cell_types))
    elif args.input_model == "alexnet":
        alexnet_pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=args.pretrained)
        model = CNNModel(pretrained_model=alexnet_pretrained, num_classes=len(cell_types))
    else:
        print(f'Please provide a valid model name, the model name shall be in the list of the following models: ResNeXt50, GoogleNet, Inception_v3, vgg19, efficientnet_v2, resnet50, resnet101, alexnet')
        sys.exit(1)

    # Data augmentation pipelines
    val_transform_pipeline = albumentations.Compose([
        albumentations.Normalize(mean=(0.5642, 0.5026, 0.6960), std=(0.2724, 0.2838, 0.2167)),
    ])

    train_transform_pipeline = albumentations.Compose([
        albumentations.ShiftScaleRotate(p=0.8),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.VerticalFlip(p=0.5),
        albumentations.Affine(shear=(-10,10), p=0.3),
        albumentations.ISONoise(color_shift=(0.01, 0.02), intensity=(0.05, 0.01), always_apply=False, p=0.2),
        albumentations.RandomBrightnessContrast(contrast_limit=0.4, brightness_by_max=0.4, p=0.5),
        albumentations.CLAHE(p=0.3),
        albumentations.ColorJitter(p=0.2),
        albumentations.RandomGamma(p=0.2),
        albumentations.Normalize(mean=(0.5642, 0.5026, 0.6960), std=(0.2724, 0.2838, 0.2167)),
    ])

    trainer = trainer_classification(
        train_image_files=X_train,
        validation_image_files=X_val,
        model=model,
        train_img_transform=train_transform_pipeline,
        val_img_transform=val_transform_pipeline,
        init_lr=args.init_lr,
        lr_decay_every_x_epochs=args.lr_decay_every_x_epochs,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        gamma=args.gamma,
        df=cell_types_df,
        save_checkpoints_dir=args.save_checkpoints_dir
    )

    trained_model = trainer.train(model)