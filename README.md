# Traffic Forecasting with LSTM, Self-Attention and Multi-Head Attention

## Overview

This project implements a deep learning model for traffic forecasting using PyTorch.

The architecture combines:

* LSTM (Long Short-Term Memory)
* Custom Self-Attention
* Multi-Head Attention
* Fully Connected Layers

The goal is to learn temporal patterns from historical traffic values and predict the next traffic value in a sequence.

---

## Model Architecture

```text
Traffic Data
      ↓
Sliding Window Dataset
      ↓
LSTM
      ↓
Self-Attention
      ↓
Multi-Head Attention
      ↓
Linear (64 → 32)
      ↓
ReLU
      ↓
Linear (32 → 1)
      ↓
Traffic Prediction
```

---

## Components

### Dataset Preparation

The dataset converts raw traffic values into supervised learning sequences using a sliding window approach.

Example:

```text
Input  : [120, 125, 130, 128, 135]
Target : 140
```

The data is converted into PyTorch tensors with shape:

```text
(batch_size, sequence_length, features)
```

---

### LSTM Layer

The LSTM learns temporal dependencies and generates hidden representations for each time step.

```python
nn.LSTM(
    input_size=1,
    hidden_size=64,
    batch_first=True
)
```

Output shape:

```text
(batch_size, sequence_length, 64)
```

---

### Self-Attention

A custom self-attention module is implemented from scratch.

The module computes:

```text
Q = Query(X)
K = Key(X)
V = Value(X)
```

Attention scores:

```text
Score = QKᵀ / √d
```

Attention weights:

```text
Weights = Softmax(Score)
```

Output:

```text
Output = Weights × V
```

This allows the model to focus on the most important temporal features.

---

### Multi-Head Attention

After self-attention, a Multi-Head Attention layer is applied.

```python
nn.MultiheadAttention(
    embed_dim=64,
    num_heads=4,
    batch_first=True
)
```

Multiple attention heads enable the model to learn different relationships within the sequence.

---

### Fully Connected Layers

The final representation is passed through:

```text
64 → 32 → 1
```

with ReLU activation.

The final output is a single predicted traffic value.

---

## Training

Loss Function:

```python
nn.MSELoss()
```

Optimizer:

```python
torch.optim.Adam(
    model.parameters(),
    lr=0.01
)
```

Training is performed using backpropagation and gradient descent.

---

## Example Prediction

Input:

```text
[175, 180, 185, 190, 188]
```

Output:

```text
Predicted next traffic value
```

---

## Technologies Used

* Python
* PyTorch
* LSTM
* Self-Attention
* Multi-Head Attention

---

## Learning Objectives

This project was created to practice:

* Time Series Forecasting
* Sequence Modeling
* LSTM Networks
* Self-Attention Mechanisms
* Multi-Head Attention
* PyTorch Model Development

---

## Future Improvements

* Data normalization
* Validation dataset
* Early stopping
* Learning rate scheduling
* Transformer Encoder implementation
* Real-world network traffic datasets

---

## Author

Deep Learning and AI Engineering Student

Focused on:

* PyTorch
* Time Series Analysis
* Attention Mechanisms
* Transformer Models
* Network and Telecommunications Applications
