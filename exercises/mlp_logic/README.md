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

