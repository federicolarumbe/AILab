# Boolean Logic MLP

A multi-layer perceptron (MLP) that learns boolean logic operations from truth tables.

## Overview

This MLP takes as input:
- Two boolean values (0 or 1)
- An operator token (and, or, xor, imply, not)

And outputs:
- One boolean value (0 or 1)

The model is trained on truth tables for all supported boolean operators.

## Architecture

The MLP has:
- **Input layer**: 7 dimensions (2 for booleans + 5 for operator one-hot encoding)
- **Hidden layers**: `n` layers with dimension `m` (configurable)
- **Output layer**: 1 dimension (boolean output)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Training

Train with default parameters (hidden_dim=64, depth=2):

```bash
python train.py
```

### Custom Configuration

Train with custom hidden dimension `m` and depth `n`:

```bash
python train.py --hidden_dim 128 --depth 3 --epochs 2000
```

### Command Line Arguments

- `--hidden_dim`: Hidden dimension m (default: 64)
- `--depth`: Number of hidden layers n (default: 2)
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)

## Supported Operators

- **AND**: Logical AND
- **OR**: Logical OR
- **XOR**: Exclusive OR
- **IMPLY**: Logical implication (x1 -> x2)
- **NOT**: Logical NOT (applied to x1)

## Example

```python
from mlp_model import BooleanLogicMLP
from data_utils import encode_input
import torch

# Load or create model
model = BooleanLogicMLP(hidden_dim=64, depth=2)

# Encode input: x1=1, x2=0, operator='and'
input_vec = encode_input(1, 0, 'and')
input_tensor = torch.from_numpy(input_vec).unsqueeze(0)

# Get prediction
prediction = model.predict(input_tensor)
print(f"Prediction: {prediction.item()}")  # Should output 0 for (1 AND 0)
```

## Training Data

The model is trained on complete truth tables for all operators:
- Each operator has 4 combinations (2^2 for two boolean inputs)
- Total training samples: 5 operators × 4 combinations = 20 samples

---

## Sequence Model

A sequence-based MLP that handles variable-length sequences of boolean operations.

### Overview

The sequence model can process:
- **Chain sequences**: `[x1, x2, op1, x3, op2, x4, op3, ...]` representing `((x1 op1 x2) op2 x3) op3 x4) ...`
- **Stack sequences**: `[x1, x2, op1, x3, x4, op2, op3]` representing `(x1 op1 x2) op3 (x3 op2 x4)`

### Architecture

The sequence model uses:
- **RNN/LSTM layers**: `n` layers with hidden dimension `m` (configurable)
- **Input encoding**: Each token (value or operator) is encoded as 6 dimensions
- **Output layer**: MLP that produces a single boolean value

### Usage

Train the sequence model with default parameters:

```bash
python train_sequence.py
```

Custom configuration:

```bash
python train_sequence.py --hidden_dim 128 --num_layers 3 --epochs 2000 --max_length 9
```

### Command Line Arguments

- `--hidden_dim`: Hidden dimension m (default: 64)
- `--num_layers`: Number of RNN/LSTM layers n (default: 2)
- `--use_lstm`: Use LSTM (default: True)
- `--use_gru`: Use GRU instead of LSTM
- `--epochs`: Number of training epochs (default: 1000)
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Learning rate (default: 0.001)
- `--max_length`: Maximum sequence length (default: 7)
- `--num_samples`: Number of samples per length (default: 100)

### Example

```python
from sequence_model import SequenceLogicMLP
from sequence_data_utils import encode_sequence, pad_sequences
import torch

# Load or create model
model = SequenceLogicMLP(hidden_dim=64, num_layers=2, use_lstm=True)

# Chain sequence: [1, 0, 'and', 1, 'or'] represents ((1 AND 0) OR 1)
sequence = [1, 0, 'and', 1, 'or']
encoded = encode_sequence(sequence)
padded, lengths = pad_sequences([encoded])

input_tensor = torch.from_numpy(padded).float()
lengths_tensor = torch.from_numpy(lengths)

prediction = model.predict(input_tensor, lengths_tensor)
print(f"Prediction: {prediction.item()}")  # Should output 1 for ((1 AND 0) OR 1)
```

### Sequence Formats

**Chain sequences** evaluate left-to-right:
- `[x1, x2, op1]` → `x1 op1 x2`
- `[x1, x2, op1, x3, op2]` → `(x1 op1 x2) op2 x3`
- `[x1, x2, op1, x3, op2, x4, op3]` → `((x1 op1 x2) op2 x3) op3 x4`

**Stack sequences** use postfix notation:
- `[x1, x2, op1]` → `x1 op1 x2`
- `[x1, x2, op1, x3, x4, op2, op3]` → `(x1 op1 x2) op3 (x3 op2 x4)`

