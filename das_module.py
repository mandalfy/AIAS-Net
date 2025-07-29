import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DynamicAdaptiveSampling(nn.Module):
    def __init__(self, config):
        super(DynamicAdaptiveSampling, self).__init__()
        self.sampling_strategy = config.get("sampling_strategy", "adaptive")
        self.threshold = config.get("threshold", 0.5)
        self.min_samples_per_class = config.get("min_samples_per_class", 10)
        
        # Learnable parameters for adaptive sampling
        self.sampling_weights = nn.Parameter(torch.ones(1))
        
    def forward(self, features, targets):
        if not self.training:
            return features, targets
        
        batch_size, channels, height, width = features.shape
        
        # Calculate class distribution in the current batch
        class_counts = self._calculate_class_distribution(targets)
        
        # Determine sampling probabilities based on class imbalance
        sampling_probs = self._compute_sampling_probabilities(class_counts, targets)
        
        # Apply dynamic sampling
        sampled_features, sampled_targets = self._apply_sampling(features, targets, sampling_probs)
        
        return sampled_features, sampled_targets
    
    def _calculate_class_distribution(self, targets):
        """Calculate the distribution of classes in the current batch."""
        unique_classes, counts = torch.unique(targets, return_counts=True)
        class_counts = {}
        for cls, count in zip(unique_classes, counts):
            class_counts[cls.item()] = count.item()
        return class_counts
    
    def _compute_sampling_probabilities(self, class_counts, targets):
        """Compute sampling probabilities to balance class representation."""
        total_pixels = targets.numel()
        num_classes = len(class_counts)
        
        # Calculate inverse frequency weights
        sampling_probs = torch.ones_like(targets, dtype=torch.float32)
        
        for cls, count in class_counts.items():
            # Higher probability for minority classes
            class_weight = total_pixels / (num_classes * count)
            class_mask = (targets == cls)
            sampling_probs[class_mask] = class_weight
        
        # Normalize probabilities
        sampling_probs = sampling_probs / sampling_probs.sum()
        
        # Apply learnable sampling weights
        sampling_probs = sampling_probs * self.sampling_weights
        
        return sampling_probs
    
    def _apply_sampling(self, features, targets, sampling_probs):
        """Apply the computed sampling probabilities to features and targets."""
        batch_size, channels, height, width = features.shape
        
        # Flatten spatial dimensions for easier sampling
        features_flat = features.view(batch_size, channels, -1)  # [B, C, H*W]
        targets_flat = targets.view(batch_size, -1)  # [B, H*W]
        sampling_probs_flat = sampling_probs.view(batch_size, -1)  # [B, H*W]
        
        sampled_features_list = []
        sampled_targets_list = []
        
        for b in range(batch_size):
            # Sample indices based on probabilities
            num_samples = min(int(height * width * 0.8), height * width)  # Sample 80% of pixels
            
            try:
                sampled_indices = torch.multinomial(
                    sampling_probs_flat[b], 
                    num_samples, 
                    replacement=False
                )
            except RuntimeError:
                # Fallback to uniform sampling if multinomial fails
                sampled_indices = torch.randperm(height * width)[:num_samples]
            
            # Extract sampled features and targets
            sampled_features_b = features_flat[b, :, sampled_indices]  # [C, num_samples]
            sampled_targets_b = targets_flat[b, sampled_indices]  # [num_samples]
            
            sampled_features_list.append(sampled_features_b)
            sampled_targets_list.append(sampled_targets_b)
        
        # Reconstruct spatial dimensions (simplified - in practice, might need more sophisticated reconstruction)
        # For now, we'll pad/crop to maintain consistent dimensions
        max_samples = max([sf.shape[1] for sf in sampled_features_list])
        
        padded_features = []
        padded_targets = []
        
        for sf, st in zip(sampled_features_list, sampled_targets_list):
            if sf.shape[1] < max_samples:
                # Pad with zeros
                pad_size = max_samples - sf.shape[1]
                sf_padded = F.pad(sf, (0, pad_size), mode='constant', value=0)
                st_padded = F.pad(st, (0, pad_size), mode='constant', value=0)
            else:
                sf_padded = sf[:, :max_samples]
                st_padded = st[:max_samples]
            
            padded_features.append(sf_padded)
            padded_targets.append(st_padded)
        
        # Stack back into batch format
        sampled_features = torch.stack(padded_features, dim=0)  # [B, C, max_samples]
        sampled_targets = torch.stack(padded_targets, dim=0)  # [B, max_samples]
        
        # Reshape to approximate original spatial dimensions
        new_height = int(np.sqrt(max_samples))
        new_width = max_samples // new_height
        
        if new_height * new_width < max_samples:
            # Trim to fit exact dimensions
            sampled_features = sampled_features[:, :, :new_height * new_width]
            sampled_targets = sampled_targets[:, :new_height * new_width]
        
        sampled_features = sampled_features.view(batch_size, channels, new_height, new_width)
        sampled_targets = sampled_targets.view(batch_size, new_height, new_width)
        
        return sampled_features, sampled_targets

