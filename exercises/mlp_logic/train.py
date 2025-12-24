import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from mlp_model import BooleanLogicMLP
from data_utils import generate_all_training_data, create_data_loader


def train_model(hidden_dim=64, depth=2, epochs=1000, batch_size=32, learning_rate=0.001, device=None):
    """
    Train the boolean logic MLP.
    
    Args:
        hidden_dim (int): Hidden dimension m
        depth (int): Number of hidden layers n
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        device: PyTorch device (cuda or cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Model configuration: hidden_dim={hidden_dim}, depth={depth}")
    
    # Generate training data
    print("Generating training data from truth tables...")
    inputs, outputs = generate_all_training_data()
    print(f"Generated {len(inputs)} training samples")
    
    # Create data loader
    train_loader = create_data_loader(inputs, outputs, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = BooleanLogicMLP(hidden_dim=hidden_dim, depth=depth).to(device)
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_inputs, batch_outputs in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predicted_binary = (predictions > 0.5).float()
            correct += (predicted_binary == batch_outputs).sum().item()
            total += batch_outputs.size(0)
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        if (epoch + 1) % 100 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    print("\nTraining completed!")
    
    # Evaluate on all training data
    print("\nEvaluating on training data...")
    model.eval()
    with torch.no_grad():
        all_inputs_tensor = torch.from_numpy(inputs).to(device)
        all_outputs_tensor = torch.from_numpy(outputs).to(device)
        
        predictions = model(all_inputs_tensor)
        predicted_binary = (predictions > 0.5).float()
        
        correct = (predicted_binary == all_outputs_tensor).sum().item()
        total = all_outputs_tensor.size(0)
        accuracy = 100.0 * correct / total
        
        print(f"Final accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return model


def test_model(model, device=None):
    """
    Test the model on all truth table combinations.
    
    Args:
        model: Trained BooleanLogicMLP model
        device: PyTorch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from data_utils import OPERATOR_NAMES, encode_input
    
    print("\n" + "="*60)
    print("Testing model on all truth table combinations:")
    print("="*60)
    
    model.eval()
    with torch.no_grad():
        for operator in OPERATOR_NAMES:
            print(f"\n{operator.upper()} operator:")
            print("-" * 40)
            print(f"{'x1':<5} {'x2':<5} {'Expected':<10} {'Predicted':<10} {'Match':<5}")
            print("-" * 40)
            
            for x1 in [0, 1]:
                for x2 in [0, 1]:
                    # Get expected output
                    from data_utils import OPERATOR_FUNCTIONS
                    expected = OPERATOR_FUNCTIONS[operator](x1, x2)
                    
                    # Get prediction
                    input_vec = encode_input(x1, x2, operator)
                    input_tensor = torch.from_numpy(input_vec).unsqueeze(0).to(device)
                    prediction = model.predict(input_tensor).item()
                    
                    match = "✓" if int(prediction) == expected else "✗"
                    print(f"{x1:<5} {x2:<5} {expected:<10} {int(prediction):<10} {match:<5}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a boolean logic MLP')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension m (default: 64)')
    parser.add_argument('--depth', type=int, default=2, help='Number of hidden layers n (default: 2)')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    
    args = parser.parse_args()
    
    # Train model
    model = train_model(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Test model
    test_model(model)

