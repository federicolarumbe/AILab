import torch
import numpy as np
from data_utils import OPERATORS, OPERATOR_NAMES, encode_operator, OPERATOR_FUNCTIONS


def is_operator(token):
    """Check if a token is an operator."""
    return isinstance(token, str) and token in OPERATORS


def is_value(token):
    """Check if a token is a boolean value."""
    return token in [0, 1]


def encode_sequence_token(token):
    """
    Encode a single token (value or operator) in a sequence.
    
    Args:
        token: Either a boolean value (0 or 1) or operator name (str)
    
    Returns:
        Numpy array of shape (6,):
        - If value: [value, 0, 0, 0, 0, 0] (value + 5 zeros for operator)
        - If operator: [0, operator_one_hot] (0 + 5-dim one-hot)
    """
    if is_value(token):
        return np.concatenate([[float(token)], np.zeros(5)])
    elif is_operator(token):
        return np.concatenate([[0.0], encode_operator(token)])
    else:
        raise ValueError(f"Invalid token: {token}. Must be 0, 1, or operator name.")


def encode_sequence(sequence):
    """
    Encode a sequence of operations.
    
    Args:
        sequence: List of tokens, e.g., [x1, x2, op1, x3, op2, ...]
    
    Returns:
        Numpy array of shape (seq_len, 6) where each row is an encoded token
    """
    return np.array([encode_sequence_token(token) for token in sequence], dtype=np.float32)


def evaluate_chain_sequence(sequence):
    """
    Evaluate a chain sequence: [x1, x2, op1, x3, op2, x4, op3, ...]
    This represents: ((x1 op1 x2) op2 x3) op3 x4) ...
    
    Args:
        sequence: List of tokens representing a chain operation
    
    Returns:
        Boolean result (0 or 1)
    """
    if len(sequence) < 3:
        raise ValueError("Sequence must have at least 3 tokens: [x1, x2, op]")
    
    # Start with first two values
    if not is_value(sequence[0]) or not is_value(sequence[1]):
        raise ValueError("Sequence must start with two values")
    
    result = OPERATOR_FUNCTIONS[sequence[2]](sequence[0], sequence[1])
    
    # Process remaining tokens in pairs (value, operator)
    i = 3
    while i < len(sequence):
        if not is_value(sequence[i]):
            raise ValueError(f"Expected value at position {i}, got {sequence[i]}")
        if i + 1 >= len(sequence) or not is_operator(sequence[i + 1]):
            raise ValueError(f"Expected operator after value at position {i}")
        
        val = sequence[i]
        op = sequence[i + 1]
        result = OPERATOR_FUNCTIONS[op](result, val)
        i += 2
    
    return result


def evaluate_stack_sequence(sequence):
    """
    Evaluate a stack sequence: [x1, x2, op1, x3, x4, op2, op3]
    This represents: (x1 op1 x2) op3 (x3 op2 x4)
    
    The sequence alternates between values and operators, and can represent
    tree-like structures. We use a stack-based evaluation.
    
    Args:
        sequence: List of tokens representing a stack operation
    
    Returns:
        Boolean result (0 or 1)
    """
    stack = []
    
    for token in sequence:
        if is_value(token):
            stack.append(token)
        elif is_operator(token):
            if len(stack) < 2:
                raise ValueError(f"Not enough values on stack for operator {token}")
            
            # Pop two values, apply operator, push result
            x2 = stack.pop()
            x1 = stack.pop()
            result = OPERATOR_FUNCTIONS[token](x1, x2)
            stack.append(result)
        else:
            raise ValueError(f"Invalid token: {token}")
    
    if len(stack) != 1:
        raise ValueError(f"Invalid sequence: stack has {len(stack)} values at end")
    
    return stack[0]


def generate_chain_sequences(max_length=7, num_samples_per_length=None):
    """
    Generate chain sequences for training.
    
    Args:
        max_length: Maximum sequence length (must be odd: 3, 5, 7, ...)
        num_samples_per_length: Number of samples to generate per sequence length
    
    Returns:
        List of tuples: [(sequence, output), ...]
    """
    if num_samples_per_length is None:
        num_samples_per_length = 100
    
    sequences = []
    
    # Generate sequences of different lengths: 3, 5, 7, ...
    for seq_len in range(3, max_length + 1, 2):
        for _ in range(num_samples_per_length):
            # Generate random sequence
            sequence = []
            
            # Start with two values
            sequence.append(np.random.choice([0, 1]))
            sequence.append(np.random.choice([0, 1]))
            
            # Add operator-value pairs
            remaining = seq_len - 2
            for i in range(remaining):
                if i % 2 == 0:  # Operator
                    sequence.append(np.random.choice(OPERATOR_NAMES))
                else:  # Value
                    sequence.append(np.random.choice([0, 1]))
            
            # Evaluate the sequence
            #try:
            output = evaluate_chain_sequence(sequence)
            sequences.append((sequence, output))
            #except Exception as e:
            #    # Skip invalid sequences
            #    continue
    
    return sequences


def generate_stack_sequences(max_length=7, num_samples_per_length=None):
    """
    Generate stack sequences for training.
    
    Args:
        max_length: Maximum sequence length
        num_samples_per_length: Number of samples to generate per sequence length
    
    Returns:
        List of tuples: [(sequence, output), ...]
    """
    if num_samples_per_length is None:
        num_samples_per_length = 100
    
    sequences = []
    
    # Valid stack sequences have specific patterns:
    # - Must have n values and n-1 operators
    # - Examples: [x1, x2, op] (2 values, 1 op), [x1, x2, op1, x3, x4, op2, op3] (4 values, 3 ops)
    
    for num_values in range(2, (max_length + 1) // 2 + 1):
        num_ops = num_values - 1
        seq_len = num_values + num_ops
        
        if seq_len > max_length:
            continue
        
        for _ in range(num_samples_per_length):
            # Generate random sequence with correct pattern
            sequence = []
            
            # Add values and operators in a valid pattern
            # Simple pattern: all values first, then all operators (postfix notation)
            for _ in range(num_values):
                sequence.append(np.random.choice([0, 1]))
            
            for _ in range(num_ops):
                sequence.append(np.random.choice(OPERATOR_NAMES))
            
            # Evaluate the sequence
            try:
                output = evaluate_stack_sequence(sequence)
                sequences.append((sequence, output))
            except Exception as e:
                # Skip invalid sequences
                continue
    
    return sequences


def generate_all_sequence_data(max_length=7, num_samples_per_length=100):
    """
    Generate all sequence training data (chain only, stack sequences disabled).
    
    Args:
        max_length: Maximum sequence length
        num_samples_per_length: Number of samples per sequence length
    
    Returns:
        Tuple of (inputs, outputs, lengths, raw_sequences) where:
        - inputs: List of encoded sequences (variable length)
        - outputs: List of outputs
        - lengths: List of sequence lengths
        - raw_sequences: List of original sequences (for saving to file)
    """
    chain_seqs = generate_chain_sequences(max_length, num_samples_per_length)
    # Stack sequences disabled
    # stack_seqs = generate_stack_sequences(max_length, num_samples_per_length)
    
    all_sequences = chain_seqs  # Only chain sequences
    
    inputs = []
    outputs = []
    lengths = []
    raw_sequences = []
    
    for sequence, output in all_sequences:
        encoded = encode_sequence(sequence)
        inputs.append(encoded)
        outputs.append(output)
        lengths.append(len(sequence))
        raw_sequences.append(sequence)
    
    return inputs, outputs, lengths, raw_sequences


def pad_sequences(sequences, max_length=None):
    """
    Pad sequences to the same length.
    
    Args:
        sequences: List of numpy arrays of shape (seq_len, 6)
        max_length: Maximum length to pad to (if None, use max sequence length)
    
    Returns:
        Tuple of (padded_sequences, lengths) where:
        - padded_sequences: Numpy array of shape (N, max_length, 6)
        - lengths: Numpy array of actual sequence lengths
    """
    if max_length is None:
        max_length = max(seq.shape[0] for seq in sequences)
    
    padded = []
    lengths = []
    
    for seq in sequences:
        seq_len = seq.shape[0]
        lengths.append(seq_len)
        
        if seq_len < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - seq_len, 6), dtype=np.float32)
            padded_seq = np.concatenate([seq, padding], axis=0)
        else:
            padded_seq = seq[:max_length]
        
        padded.append(padded_seq)
    
    return np.array(padded, dtype=np.float32), np.array(lengths, dtype=np.int64)


def create_sequence_data_loader(inputs, outputs, lengths, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader for sequences.
    
    Args:
        inputs: List of encoded sequences (variable length)
        outputs: List of outputs
        lengths: List of sequence lengths
        batch_size: Batch size
        shuffle: Whether to shuffle
    
    Returns:
        torch.utils.data.DataLoader
    """
    # Pad sequences
    padded_inputs, lengths_array = pad_sequences(inputs)
    
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(padded_inputs),
        torch.from_numpy(np.array(outputs, dtype=np.float32).reshape(-1, 1)),
        torch.from_numpy(lengths_array)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

