from dataclasses import dataclass, field
from typing import List
from .base_config import BaseConfig

@dataclass
class FNNConfig(BaseConfig):
    # Model architecture
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "relu"
    
    # Training specific
    batch_size: int = 1000
    learning_rate: float = 0.005
    weight_decay: float = 0.0
    
    # Data specific
    normalize_features: bool = False
    
    def __post_init__(self):
        super().__post_init__()
        self.model_type = "fnn"
