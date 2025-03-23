import os
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from src.utils.dynamic_images import preprocess_dynamic_image

class DynamicImageDataset(Dataset):
    """
    Dataset class for dynamic images and their speaking/not-speaking labels.
    """
    def __init__(self, dynamic_images, labels, input_size=256, multi_scale=False, scale_range=(256, 320), transform=None):
        """
        Initialize the dataset.
        
        Args:
            dynamic_images: List of dynamic images (numpy arrays)
            labels: List of labels (0 for not-speaking, 1 for speaking)
            input_size: Size of the input image for the model
            multi_scale: Whether to use multi-scale preprocessing
            scale_range: Range for multi-scale resizing
            transform: Additional transforms to apply
        """
        self.dynamic_images = dynamic_images
        self.labels = labels
        self.input_size = input_size
        self.multi_scale = multi_scale
        self.scale_range = scale_range
        self.transform = transform
        
    def __len__(self):
        return len(self.dynamic_images)
    
    def __getitem__(self, idx):
        dynamic_image = self.dynamic_images[idx]
        label = self.labels[idx]
        
        # Preprocess dynamic image
        image_tensor = preprocess_dynamic_image(
            dynamic_image, 
            self.input_size, 
            self.multi_scale, 
            self.scale_range
        )
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image_tensor = self.transform(image_tensor)
        
        return image_tensor, torch.tensor(label, dtype=torch.long)

class FCNDataset(Dataset):
    """
    Dataset class for FCN segmentation training.
    """
    def __init__(self, dynamic_images, masks, transform=None):
        """
        Initialize the dataset.
        
        Args:
            dynamic_images: List of dynamic images (tensors)
            masks: List of segmentation masks (tensors)
            transform: Additional transforms to apply
        """
        self.dynamic_images = dynamic_images
        self.masks = masks
        self.transform = transform
        
    def __len__(self):
        return len(self.dynamic_images)
    
    def __getitem__(self, idx):
        dynamic_image = self.dynamic_images[idx]
        mask = self.masks[idx]
        
        # Apply additional transforms if provided
        if self.transform is not None:
            dynamic_image = self.transform(dynamic_image)
        
        return dynamic_image, mask

def create_dataloaders(dynamic_images, labels, batch_size=128, 
                     input_size=256, multi_scale=False, scale_range=(256, 320),
                     validation_split=0.1, shuffle=True):
    """
    Create train and validation dataloaders.
    
    Args:
        dynamic_images: List of dynamic images
        labels: List of labels
        batch_size: Batch size
        input_size: Input size for the model
        multi_scale: Whether to use multi-scale preprocessing
        scale_range: Range for multi-scale resizing
        validation_split: Fraction of data to use for validation
        shuffle: Whether to shuffle the data
        
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
    """
    # Determine the split index
    dataset_size = len(dynamic_images)
    val_size = int(dataset_size * validation_split)
    train_size = dataset_size - val_size
    
    # Create indices for train and validation sets
    indices = np.arange(dataset_size)
    if shuffle:
        np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create train and validation datasets
    train_dataset = DynamicImageDataset(
        [dynamic_images[i] for i in train_indices],
        [labels[i] for i in train_indices],
        input_size=input_size,
        multi_scale=multi_scale,
        scale_range=scale_range
    )
    
    val_dataset = DynamicImageDataset(
        [dynamic_images[i] for i in val_indices],
        [labels[i] for i in val_indices],
        input_size=input_size,
        multi_scale=False  # No multi-scale for validation
    )
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader 