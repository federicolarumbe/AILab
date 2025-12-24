import torch
import torch.nn as nn
import torch.nn.functional as F


class BooleanLogicMLP(nn.Module):
    """
    Multi-layer perceptron for boolean logic operations.
    
    Input: two boolean values (0 or 1) and an operator token (and, or, xor, imply, not)
    Output: one boolean value (0 or 1)
    
    Args:
        hidden_dim (int): Hidden dimension m
        depth (int): Number of hidden layers n
    """
    
    def __init__(self, hidden_dim=64, depth=2):
        super(BooleanLogicMLP, self).__init__()
        
        # Input encoding:
        # - 2 dimensions for boolean values (x1, x2)
        # - 5 dimensions for operator one-hot encoding (and, or, xor, imply, not)
        input_dim = 2 + 5
        
        # Output: single boolean value
        output_dim = 1
        
        # Build layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        layers.append(nn.Sigmoid())  # Sigmoid to output probability, then threshold to 0/1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, 7) where:
               - x[:, 0:2] are the two boolean values
               - x[:, 2:7] is the one-hot encoded operator
        
        Returns:
            Tensor of shape (batch_size, 1) with output boolean value
        """
        return self.network(x)
    
    def predict(self, x):
        """
        Predict boolean output (0 or 1).
        
        Args:
            x: Input tensor of shape (batch_size, 7)
        
        Returns:
            Tensor of shape (batch_size, 1) with binary predictions
        """
        with torch.no_grad():
            output = self.forward(x)
            return (output > 0.5).float()

