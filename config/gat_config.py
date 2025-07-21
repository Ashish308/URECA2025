from dataclasses import dataclass
from .base_config import BaseConfig
from pathlib import Path

@dataclass
class GATConfig(BaseConfig):
    # Model architecture
    hidden_channels: int = 128
    num_heads: int = 8
    num_layers: int = 3
    edge_dim: int = 1
    add_self_loops: bool = True
    
    # Graph data
    graph_dir: Path = None
    
    # Training specific
    batch_size: int = 1  # Usually 1 for full graph
    learning_rate: float = 0.005
    weight_decay: float = 0.0
    
    def __post_init__(self):
        super().__post_init__()
        if self.graph_dir is None:
            self.graph_dir = self.data_dir / "Graphs"
        else:
            self.graph_dir = Path(self.graph_dir)
        self.model_type = "gat"
