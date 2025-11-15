#!/usr/bin/env python3
"""
Spiral data generation, training, and testing script.
"""

import argparse
import math
import sys
import os
import csv
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


def train(dataset_path):
    """
    Train a model on the training dataset.
    
    Args:
        dataset_path: Path to the training dataset file
    """
    # TODO: Implement training logic
    # - Read training dataset
    # - Train model
    # - Plot results
    print(f"Training on dataset: {dataset_path}")
    print("TODO: Implement training logic")


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

