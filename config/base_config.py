from dataclasses import dataclass, asdict, field
import yaml
from pathlib import Path
from typing import Optional

@dataclass
class BaseConfig:
    # Data paths
    data_dir: Path = Path("data")
    features_dir: Optional[Path] = None
    labels_dir: Optional[Path] = None
    
    # Training parameters
    batch_size: int = 32
    epochs: int = 100
    learning_rate: float = 0.001
    early_stopping_patience: int = 10
    
    # Model parameters
    dropout: float = 0.2
    num_classes: int = 30
    input_size: int = 23693  # Number of genes
    
    # Experiment tracking
    experiment_name: str = "default"
    save_dir: Path = Path("experiments/results")
    seed: int = 42
    
    # Device
    device: str = "auto"  # auto, cuda, cpu, mps
    
    def __post_init__(self):
        # Convert string paths to Path objects
        self.data_dir = Path(self.data_dir)
        self.save_dir = Path(self.save_dir)
        
        if self.features_dir is None:
            self.features_dir = self.data_dir / "Features"
        else:
            self.features_dir = Path(self.features_dir)
            
        if self.labels_dir is None:
            self.labels_dir = self.data_dir / "Labels"
        else:
            self.labels_dir = Path(self.labels_dir)
    
    def save(self, path: Path):
        """Save configuration to YAML file"""
        config_dict = asdict(self)
        # Convert Path objects to strings for YAML serialization
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load(cls, path: Path):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)
