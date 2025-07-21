import torch
import torch.nn.functional as F
from torcheval.metrics import MulticlassAUPRC
import json
from pathlib import Path

class MetricsTracker:
    def __init__(self, num_classes: int, save_path: Path = None):
        self.num_classes = num_classes
        self.save_path = save_path
        self.reset()
    
    def reset(self):
        self.auprc_metric = MulticlassAUPRC(num_classes=self.num_classes)
        self.losses = []
        self.accuracies = []
    
    def update(self, outputs, labels, loss=None):
        """Update metrics with batch results"""
        probabilities = F.softmax(outputs, dim=1)
        self.auprc_metric.update(probabilities.cpu(), labels.cpu())
        
        if loss is not None:
            self.losses.append(loss.item())
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).float().mean().item()
        self.accuracies.append(accuracy)
    
    def compute(self):
        """Compute all metrics"""
        results = {
            'auprc': self.auprc_metric.compute().item(),
            'avg_loss': sum(self.losses) / len(self.losses) if self.losses else 0,
            'avg_accuracy': sum(self.accuracies) / len(self.accuracies) if self.accuracies else 0
        }
        return results
    
    def save(self, epoch: int = None):
        """Save metrics to file"""
        if self.save_path:
            results = self.compute()
            if epoch is not None:
                results['epoch'] = epoch
            
            with open(self.save_path, 'a') as f:
                json.dump(results, f)
                f.write('\n')