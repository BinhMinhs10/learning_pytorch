import torch
import numpy as np

# ====================================
#       tensors initialization
# ====================================
print("==" * 10 + " Indexing tensors " + 10 * "==")
a = torch.ones(3)  # tensor([1., 1., 1.])
print(a[1], " - ", float(a[1]))  # tensor(1.) - 1.0

# Numpy array to tensor
n = np.ones(5)
torch_tensor = torch.from_numpy(n)

# Tensor to numpy array 
np_arr = torch_tensor.cpu().detach().numpy()

""" Pass list of list to constructor"""
points = torch.tensor(
    [[4.0, 1.0],
     [5.0, 3.0],
     [2.0, 1.0]]
)

# Slicing
print(points[1:, 0])  # all row after the first, first column
print("Unsqueeze points: ", points[None])  # add dim of size 1, like unsqueeze : [points]

# Init tensor retains properties (shape, datatype)
x_ones = torch.ones_like(points)
x_rand = torch.rand_like(points, dtype=torch.float)
print(f"Random Tensor: {x_rand} \n")

x_ones[:, 1] = 0
# print(x_ones)

# Joining tensors
t1 = torch.cat([x_ones, x_ones], dim=1)
print("Joining tensors: ", t1)

# ====================================
#       Element-wise: trên các phần tử tương ứng giữa các tensor
# ====================================
# Multiplying computes element-wise product ******
print(f"tensor.add(tensor) {x_ones.add(points)} \t- tensor * tensor: {x_ones + points} \n")
print(f"tensor.sub(tensor) {x_ones.sub(points)} \t- tensor * tensor: {x_ones - points} \n")
print(f"tensor.mul(tensor) {x_ones.mul(points)} \t- tensor * tensor: {x_ones * points} \n")


# ====================================
# Matrix Multiplication
# ====================================
print(f"tensor.matmul(tensor.T): {x_ones.matmul(x_ones.T)} \n- tensor @ tensor.T: {x_ones @ x_ones.T}")


# ====================================
#  In-place operations '_' suffix are in-place x.copy_() will change x
# ====================================
x_ones.add_(5)
print(x_ones)

