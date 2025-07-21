from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

class BaseDataLoader(ABC):
    """Base class for data loaders"""
    
    def __init__(self, config):
        self.config = config
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    @abstractmethod
    def load_data(self):
        """Load train, val, and test datasets"""
        pass
    
    @abstractmethod
    def get_dataloader(self, dataset, batch_size, shuffle=False):
        """Create a DataLoader from dataset"""
        pass
    
    def get_train_loader(self, batch_size=None, shuffle=True):
        """Get training dataloader"""
        if self.train_data is None:
            self.load_data()
        batch_size = batch_size or self.config.batch_size
        return self.get_dataloader(self.train_data, batch_size, shuffle)
    
    def get_val_loader(self, batch_size=None, shuffle=False):
        """Get validation dataloader"""
        if self.val_data is None:
            self.load_data()
        batch_size = batch_size or self.config.batch_size
        return self.get_dataloader(self.val_data, batch_size, shuffle)
    
    def get_test_loader(self, batch_size=None, shuffle=False):
        """Get test dataloader"""
        if self.test_data is None:
            self.load_data()
        batch_size = batch_size or self.config.batch_size
        return self.get_dataloader(self.test_data, batch_size, shuffle)
    
    def _load_features_labels(self, split):
        """Load features and labels for a given split"""
        features_path = self.config.features_dir / f"{split}_features.npy"
        labels_path = self.config.labels_dir / f"{split}_labels.npy"
        
        print(f"Loading {split} features from: {features_path}")
        features = np.load(features_path)
        print(f"Loading {split} labels from: {labels_path}")
        labels = np.load(labels_path)
        
        return features, labels
