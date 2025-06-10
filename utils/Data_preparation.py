"""
Data preparation script for DeepHeme.
This script processes image data and creates a balanced dataset with proper train/validation/test splits.
"""

import os
import glob
import random
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict
from pathlib import Path

class DatasetPreparator:
    """
    Handles dataset preparation, balancing, and splitting.
    """
    
    def __init__(self, image_root_dir: str, metadata_dir: str = './metadata/'):
        """
        Initialize the dataset preparator.
        
        Args:
            image_root_dir (str): Root directory containing the image data
            metadata_dir (str): Directory to save metadata files
        """
        self.image_root_dir = image_root_dir
        self.metadata_dir = Path(metadata_dir)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        
    def collect_image_paths(self) -> Tuple[List[str], List[str]]:
        """
        Collect all image paths and their corresponding labels.
        
        Returns:
            Tuple[List[str], List[str]]: Lists of image paths and labels
        """
        image_paths = glob.glob(os.path.join(self.image_root_dir, '*', '*.png'))
        labels = [Path(path).parent.name for path in image_paths]
        return image_paths, labels
    
    def balance_dataset(self, image_paths: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
        """
        Balance the dataset by oversampling minority classes.
        
        Args:
            image_paths (List[str]): List of image paths
            labels (List[str]): List of corresponding labels
            
        Returns:
            Tuple[List[str], List[str]]: Balanced lists of image paths and labels
        """
        label_counter = Counter(labels)
        max_samples_per_class = max(label_counter.values())
        
        balanced_paths = []
        balanced_labels = []
        
        for label in label_counter.keys():
            class_paths = glob.glob(os.path.join(self.image_root_dir, label, '*.png'))
            current_samples = len(class_paths)
            
            if current_samples < max_samples_per_class:
                samples_needed = max_samples_per_class - current_samples
                
                if samples_needed > current_samples:
                    # Duplicate entire class if needed
                    num_complete_duplicates = samples_needed // current_samples
                    remaining_samples = samples_needed % current_samples
                    
                    # Add complete duplicates
                    for _ in range(num_complete_duplicates):
                        balanced_paths.extend(class_paths)
                        balanced_labels.extend([label] * current_samples)
                    
                    # Add remaining samples
                    balanced_paths.extend(random.sample(class_paths, remaining_samples))
                    balanced_labels.extend([label] * remaining_samples)
                
                else:
                    # Randomly sample additional images
                    balanced_paths.extend(random.sample(class_paths, samples_needed))
                    balanced_labels.extend([label] * samples_needed)
            
            # Add original samples
            balanced_paths.extend(class_paths)
            balanced_labels.extend([label] * current_samples)
            
        return balanced_paths, balanced_labels
    
    def create_train_val_test_splits(self, 
                                   image_paths: List[str], 
                                   labels: List[str], 
                                   test_ratio: float = 0.2,
                                   val_ratio: float = 0.2) -> pd.DataFrame:
        """
        Create train/validation/test splits while maintaining class distribution.
        
        Args:
            image_paths (List[str]): List of image paths
            labels (List[str]): List of corresponding labels
            test_ratio (float): Ratio of data to use for testing
            val_ratio (float): Ratio of training data to use for validation
            
        Returns:
            pd.DataFrame: DataFrame containing paths, labels, and splits
        """
        # Create initial DataFrame
        data_info = pd.DataFrame({
            'fpath': image_paths,
            'label': labels,
            'split': 'train'
        })
        
        # Create test split
        test_indices = data_info.groupby('label').sample(frac=test_ratio).index
        data_info.loc[test_indices, 'split'] = 'test'
        
        # Create validation split from remaining training data
        train_data = data_info[data_info['split'] == 'train']
        val_indices = train_data.groupby('label').sample(frac=val_ratio).index
        data_info.loc[val_indices, 'split'] = 'val'
        
        return data_info
    
    def prepare_dataset(self) -> str:
        """
        Prepare the complete dataset with balancing and splitting.
        
        Returns:
            str: Path to the created metadata file
        """
        # Collect and balance data
        image_paths, labels = self.collect_image_paths()
        balanced_paths, balanced_labels = self.balance_dataset(image_paths, labels)
        
        # Create splits
        data_info = self.create_train_val_test_splits(balanced_paths, balanced_labels)
        
        # Save metadata
        output_path = self.metadata_dir / 'data_info.csv'
        data_info.to_csv(output_path, index=False)
        
        return str(output_path)

def main():
    """
    Main execution function.
    """
    # Configuration
    image_root_dir = ''  # Set your image root directory here
    metadata_dir = './metadata/'
    
    # Prepare dataset
    preparator = DatasetPreparator(image_root_dir, metadata_dir)
    metadata_path = preparator.prepare_dataset()
    
    print(f"Dataset preparation complete. Metadata saved to: {metadata_path}")

if __name__ == "__main__":
    main()