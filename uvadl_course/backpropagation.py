"""Dyna mic Computation Graph and Backpropagation

Notes:
    Automatically get gradients/derivatives of functions
    If output have multiple variable that case we talk about gradients

Args:
    params1 (int): the first parameter

Returns:
    bool: True for success
"""

import torch
x = torch.ones((3,))
x.requires_grad_(True)
print(x.requires_grad)

# y = 1/|x| * Tong(|(x+2)^2 + 3|)
x = torch.arange(3, dtype=torch.float32, requires_grad=True)
print("x: ", x)

# Each node defined a func for calculating gradients
a = (x + 2) ** 2 + 3
y = a.mean()
print("y: ", y)

# Backpropagation on graph
y.backward()

# Gradient indicates how a change in x will affect
print(x.grad)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Device", device)
import time
x = torch.randn(5000, 5000)

## CPU version
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print("CPU time: %6.5fs" % (end_time - start_time))

## GPU version
x = x.to(device)
# The first operation on a CUDA device can be slow as it has to establish a CPU-GPU communication first.
# Hence, we run an arbitrary command first without timing it for a fair comparison.
if torch.cuda.is_available():
    _ = torch.matmul(x*0.0, x)
start_time = time.time()
_ = torch.matmul(x, x)
end_time = time.time()
print("GPU time: %6.5fs" % (end_time - start_time))
