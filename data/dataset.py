import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data.distributed import DistributedSampler

# Global transforms for consistent usage across the codebase
def get_transforms(image_size=64):
    """
    Get the input and output transforms for the CelebA dataset.
    
    Args:
        image_size (int): Target size for the images (will be resized to square)
        
    Returns:
        tuple: (input_transform, output_transform)
    """
    input_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    output_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size)
    ])
    
    return input_transform, output_transform

class CelebADataset(Dataset):
    def __init__(self, root_dir, attr_path, transform=None, target_size=64):
        """
        Args:
            root_dir (string): Directory with all the images
            attr_path (string): Path to attribute annotations
            transform (callable, optional): Optional transform to be applied on a sample
            target_size (int): Target size for the images (will be resized to square)
        """
        self.root_dir = root_dir
        self.target_size = (target_size, target_size)  # Using square images
        
        # Read attributes more efficiently
        self.attr_df = pd.read_csv(attr_path, sep='\s+', skiprows=1, memory_map=True)
        self.img_names = self.attr_df.index.values
        self.attributes = self.attr_df.columns.values
        
        # Get default transforms if none provided
        if transform is None:
            self.transform, _ = get_transforms(target_size)
        else:
            self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_name = os.path.join(self.root_dir, self.img_names[idx])
        
        # Use error handling for corrupt images
        try:
            # Use faster image loading
            image = Image.open(img_name).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Warning: Error with image {img_name}: {e}")
            image = torch.zeros(3, *self.target_size)
        
        # Convert attributes more efficiently
        attributes = torch.FloatTensor((self.attr_df.iloc[idx].values + 1) / 2)
        
        return image, attributes

def get_celeba_dataset(root_dir, attr_path, target_size=(218, 178)):
    """
    Get the CelebA dataset directly without wrapping it in a DataLoader
    
    Args:
        root_dir (string): Directory with all the images
        attr_path (string): Path to attribute annotations
        target_size (tuple): Target size for the images
    """
    return CelebADataset(root_dir=root_dir, attr_path=attr_path, target_size=target_size)

def get_celeba_dataloader(root_dir, attr_path, batch_size=32, num_workers=4, shuffle=True, 
                         distributed=False, validation_split=0.1, pin_memory=True, prefetch_factor=2, target_size=64):
    """
    Create DataLoader for CelebA dataset with support for distributed training
    
    Args:
        root_dir (string): Directory with all the images
        attr_path (string): Path to attribute annotations
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        shuffle (bool): Whether to shuffle the data
        distributed (bool): Whether to use DistributedDataParallel
        validation_split (float): Fraction of data to use for validation
        pin_memory (bool): Whether to pin memory for faster GPU transfer
        prefetch_factor (int): Number of batches loaded in advance by each worker
        target_size (int): Target size for the images (will be resized to square)
    """
    dataset = CelebADataset(root_dir=root_dir, attr_path=attr_path, target_size=target_size)
    
    # Split dataset into train and validation
    val_size = int(validation_split * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create samplers for distributed training
    train_sampler = DistributedSampler(train_dataset) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    
    # Adjust batch size for distributed training
    if distributed:
        batch_size = batch_size // torch.distributed.get_world_size()
    
    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None) and shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),  # Only pin memory for CUDA
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,  # Larger batch size for validation
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),  # Only pin memory for CUDA
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader 