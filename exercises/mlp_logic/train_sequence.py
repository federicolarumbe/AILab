import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import json
import os
import numpy as np
from sequence_model import SequenceLogicMLP
from sequence_data_utils import (
    generate_all_sequence_data,
    create_sequence_data_loader,
    evaluate_chain_sequence,
    encode_sequence
)

# ANSI color codes
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def generate_simple_chain_test_cases():
    """
    Generate all test cases for simple chain operations [x1, x2, op].
    Includes all combinations for AND, OR, XOR, IMPLY operators.
    
    Returns:
        List of tuples: [(sequence, description), ...]
    """
    test_cases = []
    operators = ['and', 'or', 'xor', 'imply']
    
    # Generate all combinations: 4 operators × 2×2 values = 16 test cases
    for op in operators:
        for x1 in [0, 1]:
            for x2 in [0, 1]:
                test_cases.append((
                    [x1, x2, op],
                    f"Simple {op.upper()}: {x1} {op.upper()} {x2}"
                ))
    
    # Also add NOT operator test cases (NOT is unary, uses x1, ignores x2)
    for x1 in [0, 1]:
        test_cases.append((
            [x1, 0, 'not'],  # x2 is ignored for NOT
            f"Simple NOT: NOT {x1}"
        ))
    
    return test_cases


def load_or_generate_training_data(training_file=None, max_length=7, num_samples_per_length=100):
    """
    Load training data from JSON file or generate it.
    
    Args:
        training_file: Path to JSON file with training data (if None, generate new data)
        max_length: Maximum sequence length for generation (only used if training_file is None)
        num_samples_per_length: Number of samples per length for generation (only used if training_file is None)
    
    Returns:
        Tuple of (inputs, outputs, lengths, raw_sequences) where:
        - inputs: List of encoded sequences (variable length)
        - outputs: List of outputs
        - lengths: List of sequence lengths
        - raw_sequences: List of original sequences
    """
    if training_file:
        # Load from JSON file
        print(f"Loading training data from {training_file}...")
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        raw_sequences = []
        outputs = []
        lengths = []
        
        for sample in training_data['samples']:
            raw_sequences.append(sample['sequence'])
            outputs.append(sample['output'])
            lengths.append(sample['length'])
        
        # Encode sequences
        inputs = [encode_sequence(seq) for seq in raw_sequences]
        
        # Convert outputs and lengths to numpy arrays for consistency
        outputs = np.array(outputs, dtype=np.float32)
        lengths = np.array(lengths, dtype=np.int64)
        
        print(f"Loaded {len(inputs)} training samples from file")
        if len(lengths) > 0:
            print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        
        return inputs, outputs, lengths, raw_sequences
    else:
        # Generate new training data
        print("Generating sequence training data...")
        inputs, outputs, lengths, raw_sequences = generate_all_sequence_data(
            max_length=max_length,
            num_samples_per_length=num_samples_per_length
        )
        print(f"Generated {len(inputs)} training samples")
        if len(lengths) > 0:
            print(f"Sequence lengths: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)/len(lengths):.1f}")
        else:
            print("Warning: No training samples generated!")
            return None, None, None, None
        
        # Save training set to file
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_python_type(obj):
            """Convert numpy types to native Python types."""
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [convert_to_python_type(x) for x in obj]
            elif isinstance(obj, list):
                return [convert_to_python_type(x) for x in obj]
            else:
                return obj
        
        training_data = {
            'samples': [
                {
                    'sequence': [convert_to_python_type(x) for x in sequence],
                    'output': convert_to_python_type(output),
                    'length': convert_to_python_type(length)
                }
                for sequence, output, length in zip(raw_sequences, outputs, lengths)
            ],
            'metadata': {
                'total_samples': len(inputs),
                'max_length': int(max_length),
                'num_samples_per_length': int(num_samples_per_length),
                'min_length': convert_to_python_type(min(lengths)) if lengths else 0,
                'max_seq_length': convert_to_python_type(max(lengths)) if lengths else 0,
                'avg_length': convert_to_python_type(sum(lengths) / len(lengths)) if lengths else 0.0
            }
        }
        
        output_file = '/tmp/sequence_training_set.json'
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"Training set saved to {output_file}")
        
        return inputs, outputs, lengths, raw_sequences


def train_model(hidden_dim=64, num_layers=2, use_lstm=True, epochs=1000, 
                batch_size=32, learning_rate=0.001, max_length=7, 
                num_samples_per_length=100, training_file=None, device=None):
    """
    Train the sequence boolean logic MLP.
    
    Args:
        hidden_dim (int): Hidden dimension m
        num_layers (int): Number of RNN/LSTM layers n
        use_lstm (bool): Use LSTM if True, GRU if False
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate
        max_length (int): Maximum sequence length for training data (only used if training_file is None)
        num_samples_per_length (int): Number of samples per sequence length (only used if training_file is None)
        training_file (str): Path to JSON file with training data (if provided, loads instead of generating)
        device: PyTorch device (cuda or cpu)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Model configuration: hidden_dim={hidden_dim}, num_layers={num_layers}, use_lstm={use_lstm}")
    
    # Load or generate training data
    inputs, outputs, lengths, raw_sequences = load_or_generate_training_data(
        training_file=training_file,
        max_length=max_length,
        num_samples_per_length=num_samples_per_length
    )
    
    if inputs is None:
        return None
    
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
    
    # Count and print model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
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
    
    # Generate all simple operation test cases (all combinations)
    chain_examples = generate_simple_chain_test_cases()
    
    # Add test cases with 3 values: [x1, x2, op1, x3, op2] representing (x1 op1 x2) op2 x3
    chain_examples.extend([
        ([1, 1, 'or', 0, 'and'], "Chain 3: (1 OR 1) AND 0"),
        ([0, 0, 'and', 1, 'or'], "Chain 3: (0 AND 0) OR 1"),
        ([0, 1, 'xor', 1, 'and'], "Chain 3: (0 XOR 1) AND 1"),
        ([1, 0, 'or', 0, 'xor'], "Chain 3: (1 OR 0) XOR 0"),
        ([0, 1, 'imply', 1, 'and'], "Chain 3: (0 IMPLY 1) AND 1"),
    ])
    
    # Add test cases with 4 values: [x1, x2, op1, x3, op2, x4, op3] representing ((x1 op1 x2) op2 x3) op3 x4
    chain_examples.extend([
        ([0, 1, 'xor', 1, 'and', 0, 'or'], "Chain 4: ((0 XOR 1) AND 1) OR 0"),
        ([0, 0, 'and', 1, 'or', 1, 'and'], "Chain 4: ((0 AND 0) OR 1) AND 1"),
        ([1, 0, 'or', 0, 'and', 1, 'xor'], "Chain 4: ((1 OR 0) AND 0) XOR 1"),
        ([0, 1, 'imply', 1, 'and', 0, 'or'], "Chain 4: ((0 IMPLY 1) AND 1) OR 0"),
        ([1, 1, 'and', 0, 'or', 1, 'xor'], "Chain 4: ((1 AND 1) OR 0) XOR 1"),
    ])
    
    # Add test cases with 5 values: [x1, x2, op1, x3, op2, x4, op3, x5, op4] representing (((x1 op1 x2) op2 x3) op3 x4) op4 x5
    chain_examples.extend([
        ([1, 1, 'and', 0, 'or', 1, 'and', 0, 'or'], "Chain 5: (((1 AND 1) OR 0) AND 1) OR 0"),
        ([0, 0, 'or', 1, 'and', 0, 'xor', 1, 'or'], "Chain 5: (((0 OR 0) AND 1) XOR 0) OR 1"),
        ([0, 1, 'xor', 1, 'and', 0, 'or', 1, 'and'], "Chain 5: (((0 XOR 1) AND 1) OR 0) AND 1"),
        ([1, 0, 'or', 0, 'and', 1, 'xor', 0, 'or'], "Chain 5: (((1 OR 0) AND 0) XOR 1) OR 0"),
        ([0, 1, 'imply', 1, 'and', 0, 'or', 1, 'xor'], "Chain 5: (((0 IMPLY 1) AND 1) OR 0) XOR 1"),
        ([1, 0, 'and', 1, 'or', 1, 'and', 0, 'or'], "Chain 5: (((1 AND 0) OR 1) AND 1) OR 0"),
        ([1, 0, 'and', 1, 'or', 1, 'and', 0, 'and'], "Chain 5: (((1 AND 0) OR 1) AND 1) AND 0"),
    ])
    
    # Add test cases with 6 values: [x1, x2, op1, x3, op2, x4, op3, x5, op4, x6, op5] representing ((((x1 op1 x2) op2 x3) op3 x4) op4 x5) op5 x6
    chain_examples.extend([
        ([1, 1, 'and', 0, 'or', 1, 'and', 0, 'or', 1, 'xor'], "Chain 6: ((((1 AND 1) OR 0) AND 1) OR 0) XOR 1"),
        ([0, 0, 'or', 1, 'and', 0, 'xor', 1, 'or', 0, 'and'], "Chain 6: ((((0 OR 0) AND 1) XOR 0) OR 1) AND 0"),
        ([0, 1, 'xor', 1, 'and', 0, 'or', 1, 'and', 1, 'or'], "Chain 6: ((((0 XOR 1) AND 1) OR 0) AND 1) OR 1"),
        ([1, 0, 'or', 0, 'and', 1, 'xor', 0, 'or', 1, 'and'], "Chain 6: ((((1 OR 0) AND 0) XOR 1) OR 0) AND 1"),
        ([0, 1, 'imply', 1, 'and', 0, 'or', 1, 'xor', 0, 'or'], "Chain 6: ((((0 IMPLY 1) AND 1) OR 0) XOR 1) OR 0"),
    ])
    
    # Add test cases with 20 values: [x1, x2, op1, x3, op2, ..., x20, op19]
    # Two expressions that are identical except for the last value
    # Pattern: [x1, x2, op1, x3, op2, x4, op3, ..., x20, op19]
    # For 20 values: 20 values + 19 operators = 39 elements total, ending with operator
    chain_20_base = [1, 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or', 0, 'and', 1, 'or']  # 19 values + 19 operators = 38 elements, needs 1 more value and 1 more operator
    chain_examples.extend([
        (chain_20_base + [1, 'and'], "Chain 20: (last value = 1)"),
        (chain_20_base + [0, 'and'], "Chain 20: (last value = 0)"),
    ])
    
    model.eval()
    chain_correct = 0
    chain_total = 0
    
    with torch.no_grad():
        for sequence, description in chain_examples:
            try:
                expected = evaluate_chain_sequence(sequence)
                encoded = encode_sequence(sequence)
                padded, lengths = pad_sequences([encoded])
                
                input_tensor = torch.from_numpy(padded).float().to(device)
                lengths_tensor = torch.from_numpy(lengths).to(device)
                
                prediction = model.predict(input_tensor, lengths_tensor).item()
                predicted_int = int(prediction)
                is_correct = predicted_int == expected
                match = "✓" if is_correct else "✗"
                
                if is_correct:
                    chain_correct += 1
                chain_total += 1
                
                seq_str = ' '.join(str(x) for x in sequence)
                output_line = f"{description:40} | Seq: {seq_str:35} | Expected: {expected} | Predicted: {predicted_int} {match}"
                
                # Print in red if failed, normal color if passed
                if is_correct:
                    print(output_line)
                else:
                    print(f"{RED}{output_line}{RESET}")
            except Exception as e:
                print(f"Error with {description}: {e}")
                chain_total += 1
    
    chain_accuracy = 100.0 * chain_correct / chain_total if chain_total > 0 else 0.0
    print(f"\nChain sequences accuracy: {chain_accuracy:.2f}% ({chain_correct}/{chain_total})")
    
    # Stack sequences disabled
    # Overall accuracy (only chain sequences)
    print(f"\n{'='*60}")
    print(f"Overall testing accuracy: {chain_accuracy:.2f}% ({chain_correct}/{chain_total})")
    print(f"{'='*60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a sequence boolean logic MLP')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension m (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of RNN/LSTM layers n (default: 2)')
    parser.add_argument('--use_lstm', action='store_true', default=True, help='Use LSTM (default: True)')
    parser.add_argument('--use_gru', action='store_true', help='Use GRU instead of LSTM')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs (default: 500)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--max_length', type=int, default=7, help='Maximum sequence length (default: 7)')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of samples per length (default: 100)')
    parser.add_argument('--training_file', type=str, default=None, help='Path to JSON file with training data (if provided, loads instead of generating)')
    
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
        num_samples_per_length=args.num_samples,
        training_file=args.training_file
    )
    
    # Test model
    test_model(model)

