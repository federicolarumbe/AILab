import torch
import torch.nn as nn
import torch.nn.functional as F


class SequenceLogicMLP(nn.Module):
    """
    Sequence-based MLP for boolean logic operations.
    
    Handles variable-length sequences of operations:
    - Chain: [x1, x2, op1, x3, op2, ...] representing ((x1 op1 x2) op2 x3) ...
    - Stack: [x1, x2, op1, x3, x4, op2, op3] representing (x1 op1 x2) op3 (x3 op2 x4)
    
    Args:
        hidden_dim (int): Hidden dimension m for RNN/LSTM
        num_layers (int): Number of RNN/LSTM layers n
        use_lstm (bool): If True, use LSTM; if False, use GRU
    """
    
    def __init__(self, hidden_dim=64, num_layers=2, use_lstm=True):
        super(SequenceLogicMLP, self).__init__()
        
        # Input dimension: 6 (1 for value + 5 for operator one-hot)
        input_dim = 6
        
        # Output: single boolean value
        output_dim = 1
        
        # RNN/LSTM layer
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=False
            )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x, lengths=None):
        """
        Forward pass.
        
        Args:
            x: Tensor of shape (batch_size, seq_len, 6) - padded sequences
            lengths: Optional tensor of shape (batch_size,) with actual sequence lengths
        
        Returns:
            Tensor of shape (batch_size, 1) with output boolean value
        """
        # Pack sequences if lengths are provided (for efficient processing)
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
        
        # RNN forward pass
        rnn_out, hidden = self.rnn(x)
        
        # Unpack if packed
        if lengths is not None:
            rnn_out, _ = nn.utils.rnn.pad_packed_sequence(
                rnn_out, batch_first=True
            )
        
        # Use the last hidden state (or last output for each sequence)
        if lengths is not None:
            # Get the last valid output for each sequence
            batch_size = rnn_out.size(0)
            last_outputs = []
            for i in range(batch_size):
                last_idx = lengths[i].item() - 1
                last_outputs.append(rnn_out[i, last_idx, :])
            final_hidden = torch.stack(last_outputs, dim=0)
        else:
            # Use the last output
            final_hidden = rnn_out[:, -1, :]
        
        # Apply output layer
        output = self.output_layer(final_hidden)
        return output
    
    def predict(self, x, lengths=None):
        """
        Predict boolean output (0 or 1).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, 6)
            lengths: Optional tensor of actual sequence lengths
        
        Returns:
            Tensor of shape (batch_size, 1) with binary predictions
        """
        with torch.no_grad():
            output = self.forward(x, lengths)
            return (output > 0.5).float()

