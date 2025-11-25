import numpy as np
import torch

class DataNormalizer:
    """
    A class to normalize CIFAR-10 data from [0, 255] to [-1, 1] and unnormalize back.
    Works with both numpy arrays and torch tensors.
    """
    
    def __init__(self, data_min=0.0, data_max=255.0, target_min=-1.0, target_max=1.0):
        """
        Initialize the normalizer.
        
        Args:
            data_min: Minimum value of input data (default: 0.0 for uint8 images)
            data_max: Maximum value of input data (default: 255.0 for uint8 images)
            target_min: Target minimum value after normalization (default: -1.0)
            target_max: Target maximum value after normalization (default: 1.0)
        """
        self.data_min = data_min
        self.data_max = data_max
        self.target_min = target_min
        self.target_max = target_max
        
        # Compute normalization parameters
        self.data_range = data_max - data_min
        self.target_range = target_max - target_min
        self.scale = self.target_range / self.data_range
        self.offset = target_min - (data_min * self.scale)
    
    def normalize(self, data):
        """
        Normalize data from [data_min, data_max] to [target_min, target_max].
        
        Args:
            data: Input data (numpy array or torch tensor) in range [0, 255]
            
        Returns:
            Normalized data in range [-1, 1]
        """
        is_torch = isinstance(data, torch.Tensor)
        
        if is_torch:
            data = data.float()
            normalized = data * self.scale + self.offset
        else:
            data = data.astype(np.float32)
            normalized = data * self.scale + self.offset
        
        return normalized
    
    def unnormalize(self, normalized_data):
        """
        Unnormalize data from [target_min, target_max] back to [data_min, data_max].
        
        Args:
            normalized_data: Normalized data (numpy array or torch tensor) in range [-1, 1]
            
        Returns:
            Unnormalized data in range [0, 255]
        """
        is_torch = isinstance(normalized_data, torch.Tensor)
        
        if is_torch:
            unnormalized = (normalized_data - self.offset) / self.scale
            # Clamp to valid range and convert to uint8 if needed
            unnormalized = torch.clamp(unnormalized, self.data_min, self.data_max)
        else:
            unnormalized = (normalized_data - self.offset) / self.scale
            # Clamp to valid range
            unnormalized = np.clip(unnormalized, self.data_min, self.data_max)
        
        return unnormalized
    
    def __call__(self, data):
        """Allow the class to be called directly for normalization."""
        return self.normalize(data)