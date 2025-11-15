#!/usr/bin/env python3
"""
Spiral data generation, training, and testing script.
"""

import argparse
import math
import sys
import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


def gen(angle):
    """
    Generate a data point on a spiral.
    
    Args:
        angle: A real number representing an angle (1 being one turn)
    
    Returns:
        tuple: (x, y) coordinates on the plane, centered around [0, 0]
    """
    # Convert angle to radians (1 turn = 2π radians)
    theta = 2 * math.pi * angle
    
    # Spiral radius increases with angle
    # Using a simple linear increase: radius = angle * scale_factor
    radius = angle * 20  # Adjust this to control spiral tightness
    
    # Calculate x, y coordinates centered around [0, 0]
    x = radius * math.cos(theta)
    y = radius * math.sin(theta)
    
    return (x, y)


def gen_and_plot(num_points=100):
    """
    Generate multiple data points on a spiral and plot them.
    
    Args:
        num_points: Number of data points to generate (default: 100)
    """
    # Generate angles evenly spaced
    angles = [i / num_points * 2 for i in range(num_points)]  # 0 to 2 turns
    
    # Generate all points
    points = [gen(angle) for angle in angles]
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    # Save data to file
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    output_file = os.path.join(data_dir, f"spiral_data_{num_points}.csv")
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['angle', 'x', 'y'])  # Header
        for angle, x, y in zip(angles, x_coords, y_coords):
            writer.writerow([angle, x, y])
    
    print(f"Data saved to {output_file}")
    
    # Plot the spiral
    plt.figure(figsize=(8, 8))
    plt.scatter(x_coords, y_coords, s=20, alpha=0.6)
    plt.plot(x_coords, y_coords, 'b-', alpha=0.3, linewidth=0.5)
    plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Spiral Data Points (n={num_points})')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.show()


# Hidden dimension constant
HIDDEN_DIM = 10


class SpiralDataset(Dataset):
    """Dataset for spiral data."""
    
    def __init__(self, csv_path):
        """
        Args:
            csv_path: Path to CSV file with columns: angle, x, y
        """
        data = []
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    'angle': float(row['angle']),
                    'x': float(row['x']),
                    'y': float(row['y'])
                })
        
        self.angles = torch.tensor([d['angle'] for d in data], dtype=torch.float32).unsqueeze(1)
        self.targets = torch.tensor([[d['x'], d['y']] for d in data], dtype=torch.float32)
    
    def __len__(self):
        return len(self.angles)
    
    def __getitem__(self, idx):
        return self.angles[idx], self.targets[idx]


class FExpertsNetwork(nn.Module):
    """
    f_experts network that takes theta (angle) and outputs 10 candidate (x, y) points.
    
    Architecture:
    - Input: theta (shape [batch_size, 1])
    - Output: experts (shape [batch_size, 10, 2])
    - 2-3 fully-connected layers with hidden size 16-32
    - Final layer outputs 10 * 2 = 20 units, then reshape to [batch_size, 10, 2]
    """
    
    def __init__(self, hidden_dim=HIDDEN_DIM):
        super(FExpertsNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 2-3 fully-connected layers with hidden size 16-32
        self.fc1 = nn.Linear(1, 24)  # Input: 1 (theta), Output: 24
        self.fc2 = nn.Linear(24, 32)  # Hidden layer
        self.fc3 = nn.Linear(32, hidden_dim * 2)  # Output: 10 * 2 = 20
        
        self.relu = nn.ReLU()
    
    def forward(self, theta):
        """
        Forward pass.
        
        Args:
            theta: Input angles, shape [batch_size, 1]
        
        Returns:
            experts: Output candidate points, shape [batch_size, 10, 2]
        """
        x = self.relu(self.fc1(theta))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        
        # Reshape to [batch_size, 10, 2]
        experts = x.view(-1, self.hidden_dim, 2)
        
        return experts


def train(dataset_path):
    """
    Train a model on the training dataset.
    
    Args:
        dataset_path: Path to the training dataset file
    """
    print(f"Training on dataset: {dataset_path}")
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset file not found: {dataset_path}")
        return
    
    # Load dataset
    dataset = SpiralDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Initialize model
    model = FExpertsNetwork(hidden_dim=HIDDEN_DIM)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 1000
    print_every = 100
    
    # Training loop
    model.train()
    losses = []
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Model architecture: {model}")
    print(f"Hidden dimension: {HIDDEN_DIM}")
    print(f"Dataset size: {len(dataset)}")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for theta, target in dataloader:
            optimizer.zero_grad()
            
            # Forward pass: get experts [batch_size, 10, 2]
            experts = model(theta)
            
            # For loss, we need to select the best expert or use all experts
            # For now, let's use the first expert (index 0) as the prediction
            # In a more sophisticated setup, you might want to use the closest expert
            prediction = experts[:, 0, :]  # [batch_size, 2]
            
            # Compute loss
            loss = criterion(prediction, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0
        losses.append(avg_loss)
        
        if (epoch + 1) % print_every == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print(f"Training completed. Final loss: {losses[-1]:.6f}")
    
    # Save model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "spiral_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()
    
    # Plot predictions vs ground truth
    model.eval()
    with torch.no_grad():
        all_theta = dataset.angles
        all_targets = dataset.targets
        all_experts = model(all_theta)
        predictions = all_experts[:, 0, :].numpy()  # Use first expert
        
        targets_np = all_targets.numpy()
        
        plt.figure(figsize=(12, 5))
        
        # Plot 1: Ground truth
        plt.subplot(1, 2, 1)
        plt.scatter(targets_np[:, 0], targets_np[:, 1], s=20, alpha=0.6, c='blue', label='Ground Truth')
        plt.plot(targets_np[:, 0], targets_np[:, 1], 'b-', alpha=0.3, linewidth=0.5)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ground Truth Spiral')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        # Plot 2: Predictions
        plt.subplot(1, 2, 2)
        plt.scatter(predictions[:, 0], predictions[:, 1], s=20, alpha=0.6, c='red', label='Predictions')
        plt.plot(predictions[:, 0], predictions[:, 1], 'r-', alpha=0.3, linewidth=0.5)
        plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Predicted Spiral')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Plot 3: All experts for a few samples
        plt.figure(figsize=(10, 8))
        sample_indices = [0, len(dataset)//4, len(dataset)//2, 3*len(dataset)//4, len(dataset)-1]
        
        for idx, sample_idx in enumerate(sample_indices):
            plt.subplot(2, 3, idx + 1)
            theta_val = all_theta[sample_idx].item()
            target_val = all_targets[sample_idx].numpy()
            experts_val = all_experts[sample_idx].numpy()  # [10, 2]
            
            # Plot all experts
            plt.scatter(experts_val[:, 0], experts_val[:, 1], s=50, alpha=0.7, c='orange', label='Experts')
            # Plot target
            plt.scatter(target_val[0], target_val[1], s=100, c='red', marker='*', label='Target', zorder=5)
            # Plot first expert (prediction)
            plt.scatter(experts_val[0, 0], experts_val[0, 1], s=100, c='blue', marker='x', label='Prediction', zorder=5)
            
            plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
            plt.axvline(x=0, color='k', linestyle='--', linewidth=0.5, alpha=0.3)
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.title(f'θ={theta_val:.3f}')
            plt.grid(True, alpha=0.3)
            plt.legend(fontsize=8)
            plt.axis('equal')
        
        plt.tight_layout()
        plt.show()


def test(dataset_path):
    """
    Test the model on a test dataset.
    
    Args:
        dataset_path: Path to the test dataset file
    """
    # TODO: Implement testing logic
    # - Read test dataset
    # - Test model
    # - Plot results
    print(f"Testing on dataset: {dataset_path}")
    print("TODO: Implement testing logic")


def main():
    parser = argparse.ArgumentParser(
        description="Spiral data generation, training, and testing"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # gen command
    gen_parser = subparsers.add_parser("gen", help="Generate data points on a spiral and plot")
    gen_parser.add_argument(
        "-n", "--num-points",
        type=int,
        default=100,
        help="Number of data points to generate (default: 100)"
    )
    
    # train command
    train_parser = subparsers.add_parser("train", help="Train a model on training dataset")
    train_parser.add_argument(
        "dataset",
        type=str,
        help="Path to training dataset file"
    )
    
    # test command
    test_parser = subparsers.add_parser("test", help="Test model on test dataset")
    test_parser.add_argument(
        "dataset",
        type=str,
        help="Path to test dataset file"
    )
    
    args = parser.parse_args()
    
    if args.command == "gen":
        gen_and_plot(args.num_points)
    elif args.command == "train":
        train(args.dataset)
    elif args.command == "test":
        test(args.dataset)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

