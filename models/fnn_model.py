import torch
import torch.nn as nn
import torch.nn.functional as F

class FNN_Model(nn.Module):
    def __init__(self, input_size=23693, num_classes=30, dropout_rate=0.3):
        super(FNN_Model, self).__init__()
              
        # Define the layers
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, num_classes)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First hidden layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Second hidden layer
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Third hidden layer
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        
        # Output layer with softmax
        x = self.fc4(x)
        # x = F.softmax(x, dim=1)
        
        return x