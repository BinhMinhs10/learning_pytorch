"""
To compute back propagation, Pytorch has a build-in differentiation engine called torch.autograd

"""
import torch

# Differentiation in Autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)

Q = 3 * a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(9 * a**2 == a.grad)
print(-2 * b == b.grad)


# frozen parameters / speed up computations doing forward pass
x = torch.rand(5, 5)
y = torch.rand(5, 5, requires_grad=True)
a = torch.add(x, y)
print(f"Dose a require gradients: {a.requires_grad}")  # True
a_det = a.detach()
print(f"Dose a require gradients: {a_det.requires_grad}")  # False

# ====================================
#  Defines the following computational graph: object of class Function
# ====================================
x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected tensor
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)

# Backward propagation step stored in grad_fn property of tensor
print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

# Computation gradients: to optimize weight of param in NN, we need compute the derivatives of loss function
loss.backward()
print(w.grad)
print(b.grad)

# ====================================
#  Optional Reading: Tensor Gradients and Jacobian Products
# ====================================
inp = torch.eye(5, requires_grad=True)
out = (inp + 1).pow(2)
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"first call \n{inp.grad}")
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Second call \n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(inp), retain_graph=True)
print(f"Call after zeroing gradient {inp.grad}")

