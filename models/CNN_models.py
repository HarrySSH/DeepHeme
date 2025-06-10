"""
CNN model architecture implementation with transfer learning support.
"""

import torch
import torch.nn as nn

class CNNModel(nn.Module):
    """
    CNN model with transfer learning capabilities.
    
    Args:
        pretrained_model: Pre-trained CNN model (e.g., ResNet, EfficientNet)
        num_classes (int): Number of output classes
    """
    
    def __init__(self, pretrained_model, num_classes=19):
        super(CNNModel, self).__init__()
        self.pretrained = pretrained_model
        self.classifier = nn.Sequential(
            nn.Linear(1000, 100),
            nn.ReLU(),
            nn.Linear(100, num_classes)
        )
        self.num_classes = num_classes
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1, num_classes)
        """
        features = self.pretrained(x)
        logits = self.classifier(features)
        predictions = torch.sigmoid(logits.reshape(logits.shape[0], 1, self.num_classes))
        return predictions