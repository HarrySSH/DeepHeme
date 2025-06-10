"""
Dataset loader for image classification tasks.
Handles image loading, preprocessing, and label encoding.
"""

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Union
import albumentations
import pandas as pd


class ImageDataset(Dataset):
    """
    Custom dataset class for image classification.
    
    Args:
        image_paths (List[str]): List of paths to image files
        split (str): Dataset split ('train', 'val', 'test', or 'compute')
        transform (Optional[albumentations.Compose]): Data augmentation pipeline
        image_size (int): Target image size
        cell_types_df (Optional[pd.DataFrame]): DataFrame containing cell type information
        encoder (Optional[object]): Custom encoder for image preprocessing
        is_external (bool): Whether the dataset is from external source
    """
    
    def __init__(self,
                 image_paths: List[str],
                 split: str = 'train',
                 transform: Optional[albumentations.Compose] = None,
                 image_size: int = 96,
                 cell_types_df: Optional[pd.DataFrame] = None,
                 encoder: Optional[object] = None,
                 is_external: bool = False):
        """
        Initialize the dataset.
        """
        super(ImageDataset, self).__init__()
        
        self.split = split
        self.transform = transform
        self.image_paths = image_paths
        self.image_size = image_size
        self.cell_types_df = cell_types_df
        self.encoder = encoder
        self.is_external = is_external

    def __len__(self) -> int:
        """
        Get the total number of samples.
        
        Returns:
            int: Number of samples in the dataset
        """
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            index (int): Index of the sample to retrieve
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - image: Preprocessed image tensor
                - label: One-hot encoded label tensor (if split != 'compute')
                - ID: Image file path
        """
        # Load and preprocess image
        image_path = self.image_paths[index]
        image = self._load_image(image_path)
        
        # Apply transformations if specified
        if self.transform is not None:
            try:
                image = self.transform(image=image)["image"]
            except Exception as e:
                raise RuntimeError(f"Error applying transformations: {str(e)}")
        
        # Resize external images
        if self.is_external:
            image = cv2.resize(
                image,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_AREA
            )
        
        # Convert image to tensor format (C, H, W)
        image = np.einsum('ijk->kij', image)
        image_tensor = torch.from_numpy(image).float()
        
        # Prepare sample dictionary
        sample = {
            "image": image_tensor,
            "ID": image_path
        }
        
        # Add label if not in compute mode
        if self.split != "compute":
            label = self._get_label(image_path)
            sample["label"] = label
        
        return sample

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load and convert image to RGB format.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.ndarray: RGB image array
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _get_label(self, image_path: str) -> torch.Tensor:
        """
        Get one-hot encoded label for the image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            torch.Tensor: One-hot encoded label tensor
        """
        cell_type = image_path.split('/')[-2]
        label_data = self.cell_types_df[
            self.cell_types_df['Cell_Types'] == cell_type
        ].iloc[:, 2:].to_numpy()
        
        return torch.from_numpy(
            label_data.reshape(1, -1)
        ).float()


def convert_probabilities_to_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert probability DataFrame to binary classification DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with cell type probabilities
        
    Returns:
        pd.DataFrame: Binary classification DataFrame
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()
    
    # Remove the 'Cell_Types' column if it exists
    if 'Cell_Types' in result_df.columns:
        result_df = result_df.drop(['Cell_Types'], axis=1)
    
    # Convert probabilities to binary classes
    result_df = result_df.apply(lambda x: x == x.max(), axis=1)
    result_df = result_df.astype(int)
    
    return result_df