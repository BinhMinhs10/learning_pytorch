import torch

# Differentiation in Autograd
a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)


Q = 3 * a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(9 * a**2 == a.grad)
print(-2 * b == b.grad)


# frozen parameters
x = torch.rand(5, 5)
y = torch.rand(5, 5)
a = x + y
print(f"Dose a require gradients: {a.requires_grad}")
a.requires_grad = True
print(f"Dose a require gradients: {a.requires_grad}")

