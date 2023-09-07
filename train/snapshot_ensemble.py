from math import pi
from math import cos

import torch
import torch.nn as nn
import sys
import time
import cv2
import os
sys.path.append("..")
from torch.utils.data import DataLoader
from Datasets.DataLoader import Img_DataLoader
from utils.utils import configure_optimizers


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


import numpy as np


def proposed_lr(initial_lr, iteration, epoch_per_cycle):
    # proposed learning late function
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2

class SnapshotEnsemble(object):
    def __init__(self, train_image_files, validation_image_files, gamma = 0.1,
                               init_lr = 0.001, weight_decay = 0.0005, batch_size = 32, epochs = 30, lr_decay_every_x_epochs = 10,
                 print_steps = 50, df = None, img_transform = False, model =False,
                save_checkpoints_dir = None, cycles = 5):
        super(SnapshotEnsemble, self).__init__()
        assert model != False, 'Please put a model!'
        assert img_transform != False, 'Please put a augumentation pipeline!'
        self.df = df
        names = list(set(self.df['Cell_Types'].tolist()))
        
        self.train_image_files = train_image_files
        self.validation_image_files = validation_image_files
        self.batch_size = batch_size
        self.epoch = epochs
        self.global_step = 0
        self.current_step = 0
        self.init_lr = init_lr
        self.lr_decay_every_x_epochs = lr_decay_every_x_epochs
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.print_steps = print_steps
        self.img_transform = img_transform
        self.model = model
        self.save_checkpoints_dir = save_checkpoints_dir
        self.cycles = cycles

    def _dataloader(self, datalist, split='train',img_transform = False):
        dataset = Img_DataLoader(img_list=datalist, split=split, transform = img_transform, df = self.df)
        shuffle = True if split == 'train' else False
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=2, shuffle=shuffle)
        return dataloader
    
    def train_one_epoch(self, epoch, train_loader, model, optimizer, lr):
        t0 = 0.0
        model.train()
        for inputs in train_loader:
            
            self.global_step += 1
            self.current_step +=1

            t1 = time.time()

            images, masks = inputs["image"].cuda(), inputs["mask"].cuda()
            mask_out = model(images)

            total_loss = nn.BCELoss()(mask_out, masks)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            t0 += (time.time() - t1)

            if self.global_step % self.print_steps == 0:
                message = "Epoch: %d Step: %d LR: %.6f Total Loss: %.4f Runtime: %.2f s/%d iters." % (epoch+1, self.global_step, lr, total_loss, t0, self.current_step)
                print("==> %s" % (message))
                self.current_step = 0
                t0 = 0.0

        return total_loss




    def val_one_epoch(self, data_loader, model, epoch):
        with torch.no_grad():
            model.eval()

            for i, inputs in enumerate(data_loader):
                images, masks = inputs["image"].cuda(), inputs["mask"].cuda()
                mask_out = model(images)
                
                if i == 0:
                    predictions = mask_out
                    groundtruths = masks
                else:
                    predictions = torch.cat((predictions, mask_out), dim=0)
                    groundtruths = torch.cat((groundtruths, masks), dim=0)

            loss = torch.nn.BCELoss()
            total_loss = loss(predictions, groundtruths)


        print("==> Epoch: %d Loss %.6f ." % (epoch+1, total_loss.cpu().numpy() ))
        torch.cuda.empty_cache()

        return total_loss
    
    def train(self, model):
        print("==> Create model.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model= nn.DataParallel(model)
        model.to(device)

        model.cuda()
        print("==> List learnable parameters")

        print("==> Load data.")
        print(len(self.train_image_files))
        print(len(self.validation_image_files))

        train_data_loader = self._dataloader(self.train_image_files, split='train', img_transform=self.img_transform)
        val_data_loader = self._dataloader(self.validation_image_files, split='val', img_transform=self.img_transform)

        print("==> Configure optimizer.")
        optimizer, _ = configure_optimizers(model, self.init_lr, self.weight_decay,
                                                       self.gamma, self.lr_decay_every_x_epochs)
        print("==> Start training")
        since = time.time()

        


        spapshots = []
        
        count = 0
        epochs_per_cycle = int(self.epoch / self.cycles)

        for i in range(self.cycles):
            loss_list = []
            for j in range(epochs_per_cycle):

                


                _epoch_loss = 0

                lr = proposed_lr(self.init_lr,
                                 j, epochs_per_cycle)
                
                optimizer.state_dict()["param_groups"][0]["lr"] = lr

                _epoch_loss = self.train_one_epoch(i*epochs_per_cycle+j, train_data_loader, model, optimizer, lr)
                _loss = self.val_one_epoch(val_data_loader, model, i*epochs_per_cycle+j)

                if _loss.detach().cpu().numpy()<= min(loss_list):
                    torch.save({'model_state_dict': model.state_dict(),}, self.save_checkpoints_dir+f'/checkpoint_se_{i}.ckpt') 
            

            # retrieve the best model
            model.load_state_dict(torch.load(self.save_checkpoints_dir+f'/checkpoint_se_{i}.ckpt')['model_state_dict'])
            spapshots.append(model.state_dict())
        return spapshots

