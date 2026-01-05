import torch
import numpy as np
from sequence_model import SequenceLogicMLP
from sequence_data_utils import encode_sequence, pad_sequences


def inspect_model_state(model, sequence, device=None):
    """
    Inspect the internal state of SequenceLogicMLP as it processes a sequence.
    
    Args:
        model: Trained SequenceLogicMLP model
        sequence: List representing the sequence (e.g., [1, 0, 'and', 1, 'or'])
        device: PyTorch device
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    
    # Encode and prepare input
    encoded = encode_sequence(sequence)
    padded, lengths = pad_sequences([encoded])
    input_tensor = torch.from_numpy(padded).float().to(device)
    lengths_tensor = torch.from_numpy(lengths).to(device)
    seq_len = lengths[0]
    
    print("="*80)
    print("MODEL STATE INSPECTOR")
    print("="*80)
    print(f"\nInput Sequence: {sequence}")
    print(f"Sequence Length: {seq_len}")
    print(f"Encoded Shape: {encoded.shape}")
    print("\n" + "="*80)
    
    # Process sequence step by step
    with torch.no_grad():
        # Get input encoding for each timestep
        print("\nINPUT ENCODING:")
        print("-" * 80)
        for i in range(seq_len):
            token = sequence[i]
            encoded_token = encoded[i]
            print(f"\nStep {i}: Token = {token}")
            print(f"  Encoded vector (6-dim): {encoded_token}")
            print(f"  - Value component: {encoded_token[0]:.3f}")
            print(f"  - Operator one-hot: {encoded_token[1:6]}")
        
        # Process through RNN step by step
        print("\n" + "="*80)
        print("RNN PROCESSING (Step by Step):")
        print("="*80)
        
        # Initialize hidden state
        hidden = None
        if model.rnn.__class__.__name__ == 'LSTM':
            h0 = torch.zeros(model.rnn.num_layers, 1, model.rnn.hidden_size).to(device)
            c0 = torch.zeros(model.rnn.num_layers, 1, model.rnn.hidden_size).to(device)
            hidden = (h0, c0)
        else:  # GRU
            hidden = torch.zeros(model.rnn.num_layers, 1, model.rnn.hidden_size).to(device)
        
        rnn_outputs = []
        
        for i in range(seq_len):
            # Get input for this timestep
            input_step = input_tensor[:, i:i+1, :]  # Shape: (1, 1, 6)
            
            # Forward through RNN
            if model.rnn.__class__.__name__ == 'LSTM':
                output_step, hidden = model.rnn(input_step, hidden)
                h_state, c_state = hidden
            else:  # GRU
                output_step, hidden = model.rnn(input_step, hidden)
                h_state = hidden
                c_state = None
            
            # Extract output (remove batch dimension)
            output_val = output_step[0, 0, :].cpu().numpy()  # Shape: (hidden_dim,)
            
            rnn_outputs.append(output_val)
            
            print(f"\n{'='*80}")
            print(f"Step {i}: Processing token '{sequence[i]}'")
            print(f"{'='*80}")
            
            # Display input
            print(f"\n  INPUT:")
            print(f"    Token: {sequence[i]}")
            print(f"    Encoded: {encoded[i]}")
            
            # Display RNN output
            print(f"\n  RNN OUTPUT:")
            print(f"    Shape: {output_val.shape}")
            print(f"    Min: {output_val.min():.6f}, Max: {output_val.max():.6f}, Mean: {output_val.mean():.6f}")
            print(f"    First 10 values: {output_val[:10]}")
            print(f"    Last 10 values: {output_val[-10:]}")
            
            # Display hidden state
            if model.rnn.__class__.__name__ == 'LSTM':
                h_np = h_state[-1, 0, :].cpu().numpy()  # Last layer, first batch
                c_np = c_state[-1, 0, :].cpu().numpy()
                print(f"\n  LSTM HIDDEN STATE (last layer):")
                print(f"    h (hidden): shape={h_np.shape}, min={h_np.min():.6f}, max={h_np.max():.6f}, mean={h_np.mean():.6f}")
                print(f"    c (cell):   shape={c_np.shape}, min={c_np.min():.6f}, max={c_np.max():.6f}, mean={c_np.mean():.6f}")
                print(f"    h first 10: {h_np[:10]}")
                print(f"    c first 10: {c_np[:10]}")
            else:  # GRU
                h_np = h_state[-1, 0, :].cpu().numpy()  # Last layer, first batch
                print(f"\n  GRU HIDDEN STATE (last layer):")
                print(f"    h (hidden): shape={h_np.shape}, min={h_np.min():.6f}, max={h_np.max():.6f}, mean={h_np.mean():.6f}")
                print(f"    h first 10: {h_np[:10]}")
                print(f"    h last 10:  {h_np[-10:]}")
        
        # Process final output through output MLP
        print("\n" + "="*80)
        print("OUTPUT MLP PROCESSING:")
        print("="*80)
        
        # Use the last RNN output
        final_rnn_output = torch.from_numpy(rnn_outputs[-1]).unsqueeze(0).to(device)  # Shape: (1, hidden_dim)
        
        print(f"\n  Input to Output MLP:")
        print(f"    Shape: {final_rnn_output.shape}")
        print(f"    Values: {final_rnn_output[0].cpu().numpy()[:10]} ...")
        
        # Process through each layer of output MLP
        x = final_rnn_output
        layer_idx = 0
        
        for name, module in model.output_layer.named_children():
            print(f"\n  Layer {layer_idx}: {name} ({module.__class__.__name__})")
            
            if isinstance(module, torch.nn.Linear):
                print(f"    Weight shape: {module.weight.shape}")
                print(f"    Bias shape: {module.bias.shape}")
                x_before = x.clone()
                x = module(x)
                print(f"    Input shape: {x_before.shape}")
                print(f"    Output shape: {x.shape}")
                print(f"    Output range: [{x.min().item():.6f}, {x.max().item():.6f}]")
                print(f"    Output mean: {x.mean().item():.6f}")
                print(f"    Output first 10: {x[0].cpu().numpy()[:10] if x.shape[1] >= 10 else x[0].cpu().numpy()}")
            elif isinstance(module, torch.nn.ReLU):
                x_before = x.clone()
                x = module(x)
                print(f"    Input range: [{x_before.min().item():.6f}, {x_before.max().item():.6f}]")
                print(f"    Output range: [{x.min().item():.6f}, {x.max().item():.6f}]")
                print(f"    Activated neurons: {(x > 0).sum().item()}/{x.numel()}")
            elif isinstance(module, torch.nn.Sigmoid):
                x_before = x.clone()
                x = module(x)
                print(f"    Input range: [{x_before.min().item():.6f}, {x_before.max().item():.6f}]")
                print(f"    Output range: [{x.min().item():.6f}, {x.max().item():.6f}]")
                print(f"    Final prediction: {x[0, 0].item():.6f}")
                print(f"    Binary prediction: {int(x[0, 0].item() > 0.5)}")
            
            layer_idx += 1
        
        # Compare with full forward pass
        print("\n" + "="*80)
        print("FULL FORWARD PASS COMPARISON:")
        print("="*80)
        full_output = model(input_tensor, lengths_tensor)
        print(f"  Full forward pass output: {full_output[0, 0].item():.6f}")
        print(f"  Step-by-step output: {x[0, 0].item():.6f}")
        print(f"  Match: {'✓' if abs(full_output[0, 0].item() - x[0, 0].item()) < 1e-5 else '✗'}")
        
        print("\n" + "="*80)


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='Inspect SequenceLogicMLP internal state')
    parser.add_argument('--model_path', type=str, help='Path to saved model (optional, will create dummy model if not provided)')
    parser.add_argument('--sequence', type=str, required=True, help='Sequence to inspect (e.g., "1,0,and,1,or")')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension (if creating dummy model)')
    parser.add_argument('--num_layers', type=int, default=2, help='Number of layers (if creating dummy model)')
    parser.add_argument('--use_lstm', action='store_true', default=True, help='Use LSTM (if creating dummy model)')
    parser.add_argument('--use_gru', action='store_true', help='Use GRU instead of LSTM')
    
    args = parser.parse_args()
    
    # Parse sequence
    seq_parts = args.sequence.split(',')
    sequence = []
    for part in seq_parts:
        part = part.strip()
        if part in ['0', '1']:
            sequence.append(int(part))
        else:
            sequence.append(part)
    
    # Load or create model
    if args.model_path:
        model = torch.load(args.model_path)
    else:
        use_lstm = not args.use_gru if args.use_gru else args.use_lstm
        model = SequenceLogicMLP(
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            use_lstm=use_lstm
        )
        print("Warning: Using untrained model. Provide --model_path for trained model.")
    
    # Inspect
    inspect_model_state(model, sequence)

