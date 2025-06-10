"""
Training module for snapshot ensemble model training.
"""

import os
import sys
import time
from math import pi, cos
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from Datasets.DataLoader import ImageDataset
from utils.utils import configure_optimizers

def cosine_learning_rate(initial_lr: float,
                        iteration: int,
                        epoch_per_cycle: int) -> float:
    """
    Compute learning rate using cosine annealing.
    
    Args:
        initial_lr: Initial learning rate
        iteration: Current iteration
        epoch_per_cycle: Number of epochs per cycle
        
    Returns:
        float: Current learning rate
    """
    return initial_lr * (cos(pi * iteration / epoch_per_cycle) + 1) / 2

class SnapshotEnsembleTrainer:
    """
    Trainer class for snapshot ensemble model training.
    """
    
    def __init__(self,
                 train_image_files: List[str],
                 validation_image_files: List[str],
                 model: nn.Module,
                 img_transform: object,
                 df: pd.DataFrame,
                 init_lr: float = 0.001,
                 weight_decay: float = 0.0005,
                 batch_size: int = 32,
                 epochs: int = 30,
                 lr_decay_every_x_epochs: int = 10,
                 gamma: float = 0.1,
                 print_steps: int = 50,
                 save_checkpoints_dir: Optional[str] = None,
                 cycles: int = 5):
        """
        Initialize the trainer.
        
        Args:
            train_image_files: List of training image paths
            validation_image_files: List of validation image paths
            model: PyTorch model to train
            img_transform: Data augmentation pipeline
            df: DataFrame containing cell type information
            init_lr: Initial learning rate
            weight_decay: Weight decay for regularization
            batch_size: Batch size for training
            epochs: Number of training epochs
            lr_decay_every_x_epochs: Learning rate decay interval
            gamma: Learning rate decay factor
            print_steps: Steps between progress prints
            save_checkpoints_dir: Directory to save model checkpoints
            cycles: Number of snapshot cycles
        """
        if not model or not img_transform:
            raise ValueError("Model and transforms are required")
            
        self.df = df
        self.train_image_files = train_image_files
        self.validation_image_files = validation_image_files
        self.batch_size = batch_size
        self.epochs = epochs
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
        
        if save_checkpoints_dir:
            os.makedirs(save_checkpoints_dir, exist_ok=True)

    def _create_dataloader(self,
                          data_list: List[str],
                          split: str = 'train',
                          img_transform: Optional[object] = None) -> DataLoader:
        """
        Create a DataLoader for the given data.
        
        Args:
            data_list: List of image paths
            split: Dataset split ('train' or 'val')
            img_transform: Data augmentation pipeline
            
        Returns:
            DataLoader: PyTorch DataLoader
        """
        dataset = ImageDataset(
            img_list=data_list,
            split=split,
            transform=img_transform,
            df=self.df
        )
        shuffle = split == 'train'
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=2,
            shuffle=shuffle
        )

    def train_one_epoch(self,
                       epoch: int,
                       train_loader: DataLoader,
                       model: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       lr: float) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
            train_loader: Training data loader
            model: Model to train
            optimizer: Optimizer
            lr: Current learning rate
            
        Returns:
            float: Average loss for the epoch
        """
        model.train()
        epoch_loss = 0.0
        batch_time = 0.0
        
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            self.global_step += 1
            self.current_step += 1
            
            start_time = time.time()
            
            # Forward pass
            images = inputs["image"].cuda()
            targets = inputs["label"].cuda()
            outputs = model(images)
            
            # Compute loss
            loss = nn.BCELoss()(outputs, targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            batch_time += time.time() - start_time
            epoch_loss += loss.item()
            
            # Print progress
            if self.global_step % self.print_steps == 0:
                avg_loss = epoch_loss / self.current_step
                avg_time = batch_time / self.current_step
                print(f"Epoch: {epoch + 1} Step: {self.global_step} "
                      f"LR: {lr:.6f} "
                      f"Loss: {avg_loss:.4f} "
                      f"Time: {avg_time:.2f}s/iter")
                self.current_step = 0
                batch_time = 0.0
        
        return epoch_loss / len(train_loader)

    def validate(self,
                data_loader: DataLoader,
                model: nn.Module,
                epoch: int) -> float:
        """
        Validate the model.
        
        Args:
            data_loader: Validation data loader
            model: Model to validate
            epoch: Current epoch number
            
        Returns:
            float: Validation loss
        """
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs in data_loader:
                images = inputs["image"].cuda()
                targets = inputs["label"].cuda()
                outputs = model(images)
                
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Concatenate all predictions and targets
        predictions = torch.cat(all_predictions, dim=0)
        targets = torch.cat(all_targets, dim=0)
        
        # Compute loss
        loss = nn.BCELoss()(predictions, targets)
        print(f"Epoch: {epoch + 1} Validation Loss: {loss.item():.6f}")
        
        torch.cuda.empty_cache()
        return loss.item()

    def train(self, model: nn.Module) -> str:
        """
        Train the model using snapshot ensemble.
        
        Args:
            model: Model to train
            
        Returns:
            str: Status message
        """
        # Setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = nn.DataParallel(model).to(device)
        
        # Create data loaders
        train_loader = self._create_dataloader(
            self.train_image_files,
            split='train',
            img_transform=self.img_transform
        )
        val_loader = self._create_dataloader(
            self.validation_image_files,
            split='val',
            img_transform=self.img_transform
        )
        
        # Configure optimizer
        optimizer, _ = configure_optimizers(
            model,
            self.init_lr,
            self.weight_decay,
            self.gamma,
            self.lr_decay_every_x_epochs
        )
        
        # Training loop
        epochs_per_cycle = int(self.epochs / self.cycles)
        snapshots = []
        
        for cycle in range(self.cycles):
            cycle_losses = []
            
            for epoch in range(epochs_per_cycle):
                current_epoch = cycle * epochs_per_cycle + epoch
                
                # Update learning rate
                lr = cosine_learning_rate(
                    self.init_lr,
                    epoch,
                    epochs_per_cycle
                )
                optimizer.param_groups[0]['lr'] = lr
                
                # Train and validate
                self.train_one_epoch(current_epoch, train_loader, model, optimizer, lr)
                val_loss = self.validate(val_loader, model, current_epoch)
                cycle_losses.append(val_loss)
                
                # Save best model in cycle
                if val_loss <= min(cycle_losses):
                    if self.save_checkpoints_dir:
                        checkpoint_path = os.path.join(
                            self.save_checkpoints_dir,
                            f'checkpoint_se_{cycle}.ckpt'
                        )
                        torch.save({
                            'model_state_dict': model.state_dict(),
                        }, checkpoint_path)
        
        return "Training completed successfully"