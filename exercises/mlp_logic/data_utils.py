import torch
import numpy as np


# Operator mapping
OPERATORS = {
    'and': 0,
    'or': 1,
    'xor': 2,
    'imply': 3,
    'not': 4
}

OPERATOR_NAMES = ['and', 'or', 'xor', 'imply', 'not']


def encode_operator(operator):
    """
    Encode operator as one-hot vector.
    
    Args:
        operator: String name of operator ('and', 'or', 'xor', 'imply', 'not')
    
    Returns:
        One-hot encoded vector of length 5
    """
    if operator not in OPERATORS:
        raise ValueError(f"Unknown operator: {operator}. Must be one of {list(OPERATORS.keys())}")
    
    one_hot = np.zeros(5)
    one_hot[OPERATORS[operator]] = 1.0
    return one_hot


def encode_input(x1, x2, operator):
    """
    Encode input as a vector.
    
    Args:
        x1: First boolean value (0 or 1)
        x2: Second boolean value (0 or 1)
        operator: Operator name ('and', 'or', 'xor', 'imply', 'not')
    
    Returns:
        Numpy array of shape (7,) containing [x1, x2, operator_one_hot]
    """
    operator_encoded = encode_operator(operator)
    return np.concatenate([[x1, x2], operator_encoded])


def boolean_and(x1, x2):
    """Logical AND operation."""
    return int(x1 and x2)


def boolean_or(x1, x2):
    """Logical OR operation."""
    return int(x1 or x2)


def boolean_xor(x1, x2):
    """Logical XOR operation."""
    return int(x1 != x2)


def boolean_imply(x1, x2):
    """Logical IMPLY operation (x1 -> x2)."""
    return int(not x1 or x2)


def boolean_not(x1, x2):
    """Logical NOT operation (applied to x1, x2 is ignored)."""
    return int(not x1)


# Operator function mapping
OPERATOR_FUNCTIONS = {
    'and': boolean_and,
    'or': boolean_or,
    'xor': boolean_xor,
    'imply': boolean_imply,
    'not': boolean_not
}


def generate_truth_table(operator):
    """
    Generate truth table for a given operator.
    
    Args:
        operator: Operator name ('and', 'or', 'xor', 'imply', 'not')
    
    Returns:
        List of tuples: [(x1, x2, operator, output), ...]
    """
    truth_table = []
    op_func = OPERATOR_FUNCTIONS[operator]
    
    # Generate all combinations of boolean inputs
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            output = op_func(x1, x2)
            truth_table.append((x1, x2, operator, output))
    
    return truth_table


def generate_all_training_data():
    """
    Generate training data from truth tables for all operators.
    
    Returns:
        Tuple of (inputs, outputs) as numpy arrays
        - inputs: shape (N, 7) where N is total number of samples
        - outputs: shape (N, 1)
    """
    all_inputs = []
    all_outputs = []
    
    for operator in OPERATOR_NAMES:
        truth_table = generate_truth_table(operator)
        for x1, x2, op, output in truth_table:
            input_vec = encode_input(x1, x2, op)
            all_inputs.append(input_vec)
            all_outputs.append([output])
    
    return np.array(all_inputs, dtype=np.float32), np.array(all_outputs, dtype=np.float32)


def create_data_loader(inputs, outputs, batch_size=32, shuffle=True):
    """
    Create a PyTorch DataLoader from numpy arrays.
    
    Args:
        inputs: Input array of shape (N, 7)
        outputs: Output array of shape (N, 1)
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
    
    Returns:
        torch.utils.data.DataLoader
    """
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(inputs),
        torch.from_numpy(outputs)
    )
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

