import torch
from .base_trainer import BaseTrainer
from models.fnn_model import FNN_Model
from dataloaders.fnn_dataloader import FNNDataLoader

class FNNTrainer(BaseTrainer):
    """Trainer for FNN model"""
    
    def _build_model(self):
        """Build FNN model"""
        return FNN_Model(
            input_size=self.config.input_size,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout,
        )
    
    def _build_dataloaders(self):
        """Build data loaders"""
        dataloader = FNNDataLoader(self.config)
        dataloader.load_data()
        
        train_loader = dataloader.get_train_loader()
        val_loader = dataloader.get_val_loader()
        test_loader = dataloader.get_test_loader()
        
        return train_loader, val_loader, test_loader
    
    def _prepare_batch(self, batch):
        """Prepare batch data for model input"""
        features, labels = batch
        return features.to(self.device), labels.to(self.device)
    
    def _forward_pass(self, inputs):
        """Forward pass through model"""
        return self.model(inputs)
