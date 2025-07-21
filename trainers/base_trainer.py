from abc import ABC, abstractmethod
import torch
from pathlib import Path
from evaluation.metrics import MetricsTracker
import logging

class BaseTrainer(ABC):
    def __init__(self, config, experiment_dir: Path):
        self.config = config
        self.experiment_dir = experiment_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model and data
        self.model = self._build_model().to(self.device)
        self.train_loader, self.val_loader, self.test_loader = self._build_dataloaders()
        
        # Setup training components
        self.optimizer = self._build_optimizer()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        # Metrics tracking
        self.train_metrics = MetricsTracker(
            config.num_classes, 
            self.experiment_dir / 'train_metrics.json'
        )
        self.val_metrics = MetricsTracker(
            config.num_classes,
            self.experiment_dir / 'val_metrics.json'
        )
        
        self.best_val_auprc = 0
        self.early_stopping_counter = 0
    
    @abstractmethod
    def _build_model(self):
        pass
    
    @abstractmethod
    def _build_dataloaders(self):
        pass
    
    def _build_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
    
    def _setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.experiment_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self):
        self.model.train()
        self.train_metrics.reset()
        
        for batch in self.train_loader:
            inputs, labels = self._prepare_batch(batch)
            
            self.optimizer.zero_grad()
            outputs = self._forward_pass(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            self.train_metrics.update(outputs, labels, loss)
        
        return self.train_metrics.compute()
    
    def validate(self):
        self.model.eval()
        self.val_metrics.reset()
        
        with torch.no_grad():
            for batch in self.val_loader:
                inputs, labels = self._prepare_batch(batch)
                outputs = self._forward_pass(inputs)
                loss = self.criterion(outputs, labels)
                self.val_metrics.update(outputs, labels, loss)
        
        return self.val_metrics.compute()
    
    @abstractmethod
    def _prepare_batch(self, batch):
        """Prepare batch data for model input"""
        pass
    
    @abstractmethod
    def _forward_pass(self, inputs):
        """Forward pass through model"""
        pass
    
    def train(self):
        self.logger.info(f"Starting training for {self.config.experiment_name}")
        
        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch()
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Train Loss: {train_metrics['avg_loss']:.4f}, "
                f"Train AUPRC: {train_metrics['auprc']:.4f}"
            )
            
            # Validate
            val_metrics = self.validate()
            self.logger.info(
                f"Epoch {epoch+1}/{self.config.epochs} - "
                f"Val Loss: {val_metrics['avg_loss']:.4f}, "
                f"Val AUPRC: {val_metrics['auprc']:.4f}"
            )
            
            # Save best model
            if val_metrics['auprc'] > self.best_val_auprc:
                self.best_val_auprc = val_metrics['auprc']
                self.save_checkpoint('model_best.pt')
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Early stopping
            if self.early_stopping_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Save final model
        self.save_checkpoint('model_final.pt')
        self.logger.info(f"Training completed. Best Val AUPRC: {self.best_val_auprc:.4f}")
    
    def save_checkpoint(self, filename):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_val_auprc': self.best_val_auprc
        }
        torch.save(checkpoint, self.experiment_dir / filename)