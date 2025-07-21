import torch
from torch.utils.data import TensorDataset, DataLoader
from .base_dataloader import BaseDataLoader

class FNNDataLoader(BaseDataLoader):
    """DataLoader for FNN model"""
    
    def load_data(self):
        """Load train, val, and test datasets"""
        # Load train data
        train_features, train_labels = self._load_features_labels('train')
        self.train_data = self._create_dataset(train_features, train_labels)
        
        # Load val data
        val_features, val_labels = self._load_features_labels('val')
        self.val_data = self._create_dataset(val_features, val_labels)
        
        # Load test data
        test_features, test_labels = self._load_features_labels('test')
        self.test_data = self._create_dataset(test_features, test_labels)
        
        print(f"Loaded datasets - Train: {len(self.train_data)}, Val: {len(self.val_data)}, Test: {len(self.test_data)}")
    
    def _create_dataset(self, features, labels):
        """Create TensorDataset from features and labels"""
        features_tensor = torch.tensor(features, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        
        if self.config.normalize_features:
            # Normalize features to zero mean and unit variance
            mean = features_tensor.mean(dim=0, keepdim=True)
            std = features_tensor.std(dim=0, keepdim=True)
            features_tensor = (features_tensor - mean) / (std + 1e-8)
        
        return TensorDataset(features_tensor, labels_tensor)
    
    def get_dataloader(self, dataset, batch_size, shuffle=False):
        """Create a DataLoader from dataset"""
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for compatibility
            pin_memory=torch.cuda.is_available()
        )
