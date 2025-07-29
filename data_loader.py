import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import json

class EarthObservationDataset(Dataset):
    def __init__(self, data_paths, config, transform=None, mode='train'):
        """
        Earth Observation Dataset for segmentation tasks.
        
        Args:
            data_paths: List of dictionaries containing 'image' and 'mask' paths
            config: Configuration object
            transform: Optional transform to be applied
            mode: 'train', 'val', or 'test'
        """
        self.data_paths = data_paths
        self.config = config
        self.transform = transform
        self.mode = mode
        self.input_size = config.data_config["input_size"]
        self.num_classes = config.data_config["num_classes"]
        
        # Define default transforms
        self.default_transform = self._get_default_transforms()
        
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, idx):
        data_item = self.data_paths[idx]
        
        # Load image and mask
        if isinstance(data_item, dict):
            image_path = data_item['image']
            mask_path = data_item['mask']
        else:
            # Assume it's a path and construct mask path
            image_path = data_item
            mask_path = data_item.replace('images', 'masks').replace('.jpg', '.png')
        
        # For prototype, generate synthetic data if paths don't exist
        if not os.path.exists(image_path):
            image, mask = self._generate_synthetic_data()
        else:
            image = self._load_image(image_path)
            mask = self._load_mask(mask_path)
        
        # Apply transforms
        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image, mask = self.default_transform(image, mask)
        
        return image, mask
    
    def _load_image(self, image_path):
        """Load and preprocess image."""
        try:
            image = Image.open(image_path).convert('RGB')
            image = image.resize(self.input_size, Image.BILINEAR)
            image = np.array(image) / 255.0
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return self._generate_synthetic_image()
    
    def _load_mask(self, mask_path):
        """Load and preprocess mask."""
        try:
            mask = Image.open(mask_path).convert('L')
            mask = mask.resize(self.input_size, Image.NEAREST)
            mask = np.array(mask)
            
            # Ensure mask values are in valid range
            mask = np.clip(mask, 0, self.num_classes - 1)
            return mask
        except Exception as e:
            print(f"Error loading mask {mask_path}: {e}")
            return self._generate_synthetic_mask()
    
    def _generate_synthetic_data(self):
        """Generate synthetic data for prototyping."""
        image = self._generate_synthetic_image()
        mask = self._generate_synthetic_mask()
        return image, mask
    
    def _generate_synthetic_image(self):
        """Generate synthetic Earth observation image."""
        height, width = self.input_size
        
        # Create a realistic-looking Earth observation image
        # Base terrain
        image = np.random.rand(height, width, 3) * 0.3 + 0.2
        
        # Add some structure (rivers, roads, etc.)
        for _ in range(5):
            start_x, start_y = np.random.randint(0, width), np.random.randint(0, height)
            end_x, end_y = np.random.randint(0, width), np.random.randint(0, height)
            
            # Draw line (simplified)
            line_thickness = np.random.randint(2, 8)
            color = np.random.rand(3) * 0.5
            
            # Simple line drawing (could be improved)
            steps = max(abs(end_x - start_x), abs(end_y - start_y))
            if steps > 0:
                for i in range(steps):
                    x = int(start_x + (end_x - start_x) * i / steps)
                    y = int(start_y + (end_y - start_y) * i / steps)
                    
                    x_min = max(0, x - line_thickness // 2)
                    x_max = min(width, x + line_thickness // 2)
                    y_min = max(0, y - line_thickness // 2)
                    y_max = min(height, y + line_thickness // 2)
                    
                    image[y_min:y_max, x_min:x_max] = color
        
        return image
    
    def _generate_synthetic_mask(self):
        """Generate synthetic segmentation mask with class imbalance."""
        height, width = self.input_size
        
        # Create imbalanced mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Add minority class regions (small patches)
        num_minority_patches = np.random.randint(1, 5)
        for _ in range(num_minority_patches):
            patch_size = np.random.randint(10, 30)
            x = np.random.randint(0, width - patch_size)
            y = np.random.randint(0, height - patch_size)
            
            # Create irregular patch
            for i in range(patch_size):
                for j in range(patch_size):
                    if np.random.rand() > 0.3:  # Make patch irregular
                        if self.num_classes > 2:
                            mask[y + i, x + j] = np.random.randint(1, self.num_classes)
                        else:
                            mask[y + i, x + j] = 1
        
        return mask
    
    def _get_default_transforms(self):
        """Get default transforms based on configuration."""
        def transform_fn(image, mask):
            # Convert to tensors
            if isinstance(image, np.ndarray):
                image = torch.from_numpy(image).permute(2, 0, 1).float()
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).long()
            
            # Normalize image
            normalize = transforms.Normalize(
                mean=self.config.data_config["normalize_mean"],
                std=self.config.data_config["normalize_std"]
            )
            image = normalize(image)
            
            # Apply augmentations during training
            if self.mode == 'train' and self.config.data_config["augmentation"]:
                image, mask = self._apply_augmentations(image, mask)
            
            return image, mask
        
        return transform_fn
    
    def _apply_augmentations(self, image, mask):
        """Apply data augmentations."""
        aug_config = self.config.data_config["augmentation"]
        
        # Horizontal flip
        if aug_config.get("horizontal_flip", False) and np.random.rand() > 0.5:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [1])
        
        # Vertical flip
        if aug_config.get("vertical_flip", False) and np.random.rand() > 0.5:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [0])
        
        # Rotation (simplified - 90 degree rotations only)
        if aug_config.get("rotation", False) and np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            image = torch.rot90(image, k, [1, 2])
            mask = torch.rot90(mask, k, [0, 1])
        
        return image, mask

def get_data_loaders(config, train_paths=None, val_paths=None, test_paths=None):
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        config: Configuration object
        train_paths: List of training data paths
        val_paths: List of validation data paths
        test_paths: List of test data paths
    
    Returns:
        Dictionary containing data loaders
    """
    # Generate synthetic paths if none provided (for prototyping)
    if train_paths is None:
        train_paths = [f"synthetic_train_{i}" for i in range(100)]
    if val_paths is None:
        val_paths = [f"synthetic_val_{i}" for i in range(20)]
    if test_paths is None:
        test_paths = [f"synthetic_test_{i}" for i in range(20)]
    
    # Create datasets
    train_dataset = EarthObservationDataset(train_paths, config, mode='train')
    val_dataset = EarthObservationDataset(val_paths, config, mode='val')
    test_dataset = EarthObservationDataset(test_paths, config, mode='test')
    
    # Create data loaders
    batch_size = config.training_config["batch_size"]
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }

def analyze_class_distribution(data_loader, num_classes):
    """
    Analyze class distribution in the dataset.
    
    Args:
        data_loader: DataLoader object
        num_classes: Number of classes
    
    Returns:
        Dictionary with class distribution statistics
    """
    class_counts = torch.zeros(num_classes)
    total_pixels = 0
    
    for _, masks in data_loader:
        for mask in masks:
            unique, counts = torch.unique(mask, return_counts=True)
            for cls, count in zip(unique, counts):
                if cls < num_classes:
                    class_counts[cls] += count
                    total_pixels += count
    
    class_frequencies = class_counts / total_pixels
    imbalance_ratio = class_counts.max() / class_counts.min() if class_counts.min() > 0 else float('inf')
    
    return {
        'class_counts': class_counts.tolist(),
        'class_frequencies': class_frequencies.tolist(),
        'total_pixels': total_pixels.item(),
        'imbalance_ratio': imbalance_ratio.item()
    }

