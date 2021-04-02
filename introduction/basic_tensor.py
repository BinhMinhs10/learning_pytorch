import torch
import numpy as np

# ====================================
#       Initialization
# ====================================
"""Example function with types documented in the docstring.

Note:
    assign values to the tensor during initialization
Methods:
    torch.zeros: create a tensor filled with zeros
    torch.ones: create a tensor filled with ones
    torch.rand: random values sampled between 0 and 1
    torch.arange: creates a tensor containing values N, N+1,..,M
    torch.Tensor: (input list) create from list element 
"""

x = np.array([[1., 2],
              [3, 4],
              [5, 6]])
t1 = torch.from_numpy(x)
# tensor with a fixed value for all element
# t2 = torch.full((3, 2), 42)
t2 = torch.rand(3, 2)
tensor = torch.arange(6)
print("Pytorch tensor: ", tensor)


# ====================================
#       Operations
# ====================================
"""Example function with types documented in the docstring.

Note:
    matrix multiplications
Methods:
    torch.matmul: matrix product over two tensors 
    torch.mm: matrix product but doesn't support broadcasting
    torch.bmm: performs support batch dim
    torch.einsum: using Einstein summation conventions 
"""

# change the shape of tensor with view(), reshape()
print("changing shape: ", tensor.view(2, 3))
print("swap dim 0 and 1: ", tensor.reshape(2, 3).permute(1, 0))

y = torch.cat((t1, t2))
print(y)

# compute sin of each element
t5 = torch.sin(y)


# ====================================
#       Indexing
# ====================================
x = torch.arange(12).view(3, 4)
# select a part of second tensor
print("Second columns: ", x[:, 1])
print("First row: ", x[0])
print("First two rows, last column: ", x[:2, -1])
