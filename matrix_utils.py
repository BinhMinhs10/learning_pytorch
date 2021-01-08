"""
torch.erfc : Computes the complementary error function of each element of input
torch.where : Choose output based on condition
torch.eig : Computes the eigenvalues and eigenvectors of a real square matrix
torch.lstsq : Computes the solution to the least squares and least norm problems
torch.svd : Singular value decomposition of an input
"""
import torch
import numpy as np


"""complementary error function."""
# a + torch.erfc(a) = 1
a = torch.randn(2, 2)
print("a: \n", a)
erfc_a = torch.erfc(a)
print("\nerfc(a): \n", erfc_a)

# example 2 - working
b = torch.randn(1, 1)
ans = torch.zeros(1, 1)
print("b: \n", b)
torch.erfc(input=b, out=ans)
print("\nerfc(b):\n", ans)

"""where condition."""
# filter tensor based on specified condition
x = torch.randn(4, 4)
y = torch.zeros(4, 4)
res = torch.where(x > 0, x, y)
print("x:\n", x)
print("\nOutput:\n", res)

"""computes eigenvalues and eigenvectors"""
# Note: square matrix
a = torch.tensor([[2., 3], [0, 4]])
l, m = torch.eig(a, eigenvectors=True)
print("a: \n",a)
print("\neigenvalues:\n", l)
print("\neigenvectors:\n", m)

"""least squares and least norm"""
# Note: A and B have same number of row
# can be used to compute weight of linear regression
a = np.array([[73, 67, 43],
              [91, 88, 64],
              [87, 134, 37],
              [102, 43, 37],
              [69, 96, 70]], dtype='float32')

b = np.array([[56, 70],
              [81, 101],
              [119, 133],
              [22, 37],
              [103, 119]], dtype='float32')
A = torch.from_numpy(a)
B = torch.from_numpy(b)

n = A.shape[1]
X, qr = torch.lstsq(B, A)
print("Least Squares solution:\n", x[:n, :])

print("\nA @ X:\n", A@X[:n, :])
print("\nB:\n", B)

# example 2 - working
A = np.array([[2, 3], [4, 2], [1, 1]], dtype='float32')
B = np.array([[1, 1], [1, 1], [2, 2]], dtype='float32')
A = torch.from_numpy(A)
B = torch.from_numpy(B)
n = A.shape[1]
print(A)
X, QR = torch.lstsq(B, A)
print("Least Squares solution:\n", X[:n, :])
print("\nQR:\n", QR)

"""compute singular value decomposition"""

A = torch.tensor([[7., 8, 9],
                  [10, 12, 15],
                  [16, 4, 8]])
print("A:\n", A)

U, S, V = torch.svd(A)
print("\nU:\n", U)
print("\nS:\n", S)
print("\nV:\n", V)

# Approx
A_approx = U @ torch.diag(S) @ V.t()
print("\nAprroximation of A:\n", A_approx)
