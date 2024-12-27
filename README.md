# Neural Networks

This repository contains an implementation of a simple neural network using a Multi-Layer Perceptron (MLP).

## Example Usage

```python
# Define the MLP with 3 input features, two hidden layers with 4 neurons each, and 1 output neuron
n = MLP(3, [4, 4, 1])

# Input data
xs = [
    [0.5, -1.5, 2.0],
    [1.0, 0.0, -1.0],
    [0.3, 1.2, -0.7],
    [-1.5, 2.0, 0.5],
    [0.7, -0.8, 1.5]
]

# True output values
y_true = [0.8, -0.5, 1.2, 0.3, -1.0]

# Training loop
for k in range(20):
    # Forward pass
    y_pred = [n(x) for x in xs]
    
    # Compute loss (Mean Squared Error)
    loss = sum((yout - ygt) ** 2 for yout, ygt in zip(y_true, y_pred))
    
    # Backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
```

## Requirements

- Python 3.x
- NumPy

## Installation

Clone the repository:

```bash
git clone https://github.com/Siddanagowda/Neural_Networks.git
```

Navigate to the project directory:

```bash
cd Neural_Networks
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Explanation
### Network Initialization:
n = MLP(3, [4, 4, 1]) initializes a Multi-Layer Perceptron with 3 input neurons, two hidden layers with 4 neurons each, and 1 output neuron.

### Input Data:
xs is a list of input vectors.
y_true is the list of true output values corresponding to the input vectors.

### Training Loop:
The loop runs for 20 iterations.

#### Forward Pass: 
Computes the predicted outputs y_pred for the input vectors.

#### Loss Calculation:
 Computes the loss as the sum of squared differences between the predicted and true outputs.

#### Backward Pass: 
Resets gradients to zero and computes the gradients of the loss with respect to the network parameters.

#### Parameter Update: 
Updates the network parameters using gradient descent with a learning rate of 0.05.

## License

This project is licensed under the MIT License.