import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from sequence_model import SequenceLogicMLP
from sequence_data_utils import (
    generate_all_sequence_data,
    create_sequence_data_loader,
    evaluate_chain_sequence,
    evaluate_stack_sequence,
    encode_sequence
)


def train_model(hidden_dim=64, num_layers=2, use_lstm=True, epochs=1000, 
                batch_size=32, learning_rate=0.001, max_length=7, 
                num_samples_per_length=100, device=None):
    """
    Train the sequence boolean logic MLP.
    
    Args:
        hidden_dim (int): Hidden dimension m
        num_layers (int): Number of RNN/LSTM layers n
        use_lstm (bool): Use LSTM if True, GRU if False
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        max_length (int): Maximum sequence length for training data
        num_samples_per_length (int): Number of samples per sequence length
        device: PyTorch device (cuda or cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Model configuration: hidden_dim={hidden_dim}, num_layers={num_layers}, use_lstm={use_lstm}")
    
    # Generate training data
    print("Generating sequence training data...")
    inputs, outputs, lengths = generate_all_sequence_data(
        max_length=max_length,
        num_samples_per_length=num_samples_per_length
    )
    print(f"Generated {len(inputs)} training samples")
    print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
    
    # Create data loader
    train_loader = create_sequence_data_loader(
        inputs, outputs, lengths, batch_size=batch_size, shuffle=True
    )
    
    # Initialize model
    model = SequenceLogicMLP(
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        use_lstm=use_lstm
    ).to(device)
    
    criterion = nn.BCELoss()  # Binary cross-entropy loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_inputs, batch_outputs, batch_lengths in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            batch_lengths = batch_lengths.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            predictions = model(batch_inputs, batch_lengths)
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
        correct = 0
        total = 0
        
        for batch_inputs, batch_outputs, batch_lengths in train_loader:
            batch_inputs = batch_inputs.to(device)
            batch_outputs = batch_outputs.to(device)
            batch_lengths = batch_lengths.to(device)
            
            predictions = model(batch_inputs, batch_lengths)
            predicted_binary = (predictions > 0.5).float()
            
            correct += (predicted_binary == batch_outputs).sum().item()
            total += batch_outputs.size(0)
        
        accuracy = 100.0 * correct / total
        print(f"Final accuracy: {accuracy:.2f}% ({correct}/{total})")
    
    return model


def test_model(model, device=None):
    """
    Test the model on example sequences.
    
    Args:
        model: Trained SequenceLogicMLP model
        device: PyTorch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    from sequence_data_utils import pad_sequences
    
    print("\n" + "="*60)
    print("Testing model on example sequences:")
    print("="*60)
    
    # Test chain sequences
    print("\nChain sequences:")
    print("-" * 60)
    chain_examples = [
        ([0, 1, 'and'], "Simple AND"),
        ([1, 1, 'or', 0, 'and'], "Chain: (1 OR 1) AND 0"),
        ([0, 1, 'xor', 1, 'and', 0, 'or'], "Chain: ((0 XOR 1) AND 1) OR 0"),
    ]
    
    model.eval()
    with torch.no_grad():
        for sequence, description in chain_examples:
            try:
                expected = evaluate_chain_sequence(sequence)
                encoded = encode_sequence(sequence)
                padded, lengths = pad_sequences([encoded])
                
                input_tensor = torch.from_numpy(padded).float().to(device)
                lengths_tensor = torch.from_numpy(lengths).to(device)
                
                prediction = model.predict(input_tensor, lengths_tensor).item()
                match = "✓" if int(prediction) == expected else "✗"
                
                seq_str = ' '.join(str(x) for x in sequence)
                print(f"{description:30} | Seq: {seq_str:30} | Expected: {expected} | Predicted: {int(prediction)} {match}")
            except Exception as e:
                print(f"Error with {description}: {e}")
    
    # Test stack sequences
    print("\nStack sequences:")
    print("-" * 60)
    stack_examples = [
        ([0, 1, 'and'], "Simple AND (stack format)"),
        ([1, 1, 'or', 0, 0, 'and', 'or'], "Stack: (1 OR 1) OR (0 AND 0)"),
        ([0, 1, 'xor', 1, 0, 'and', 'or'], "Stack: (0 XOR 1) OR (1 AND 0)"),
    ]
    
    with torch.no_grad():
        for sequence, description in stack_examples:
            try:
                expected = evaluate_stack_sequence(sequence)
                encoded = encode_sequence(sequence)
                padded, lengths = pad_sequences([encoded])
                
                input_tensor = torch.from_numpy(padded).float().to(device)
                lengths_tensor = torch.from_numpy(lengths).to(device)
                
                prediction = model.predict(input_tensor, lengths_tensor).item()
                match = "✓" if int(prediction) == expected else "✗"
                
                seq_str = ' '.join(str(x) for x in sequence)
                print(f"{description:30} | Seq: {seq_str:30} | Expected: {expected} | Predicted: {int(prediction)} {match}")
            except Exception as e:
                print(f"Error with {description}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a sequence boolean logic MLP')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension m (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN/LSTM layers n (default: 2)')
    parser.add_argument('--use_lstm', action='store_true', default=True, help='Use LSTM (default: True)')
    parser.add_argument('--use_gru', action='store_true', help='Use GRU instead of LSTM')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--max_length', type=int, default=7, help='Maximum sequence length (default: 7)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples per length (default: 100)')
    
    args = parser.parse_args()
    
    use_lstm = not args.use_gru if args.use_gru else args.use_lstm
    
    # Train model
    model = train_model(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        use_lstm=use_lstm,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        num_samples_per_length=args.num_samples
    )
    
    # Test model
    test_model(model)

