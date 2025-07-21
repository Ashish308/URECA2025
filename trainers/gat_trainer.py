import torch
from .base_trainer import BaseTrainer
from models.gat_model import GATModel
from dataloaders.gat_dataloader import GATDataLoader

class GATTrainer(BaseTrainer):
    """Trainer for GAT model"""
    
    def _build_model(self):
        """Build GAT model"""
        return GATModel(
            in_channels=self.config.input_size,
            hidden_channels=self.config.hidden_channels,
            out_channels=self.config.num_classes,
            num_heads=self.config.num_heads,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        )
    
    def _build_dataloaders(self):
        """Build data loaders"""
        dataloader = GATDataLoader(self.config)
        dataloader.load_data()
        
        train_loader = dataloader.get_train_loader()
        val_loader = dataloader.get_val_loader()
        test_loader = dataloader.get_test_loader()
        
        return train_loader, val_loader, test_loader
    
    def _prepare_batch(self, batch):
        """Prepare batch data for model input"""
        return batch.to(self.device), batch.y.to(self.device)
    
    def _forward_pass(self, inputs):
        """Forward pass through model"""
        batch = inputs
        return self.model(batch.x, batch.edge_index)
