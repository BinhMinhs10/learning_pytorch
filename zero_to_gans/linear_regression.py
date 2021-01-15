import numpy as np
import torch

inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')
# nawng suat cac loaij quar
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
# weight and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)


def model(x):
    return x @ w.t() + b


preds = model(inputs)
# big difference between prediction and actual target
print(preds)


# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    # sum all element / number of element in tensor
    return torch.sum(diff * diff) / diff.numel()

loss = mse(preds, targets)
print(loss)

# compute gradients
loss.backward()
# gradients stored in .grad
print(w.grad)

# using no_grad to indicate pytorch shouldn't track gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    # reset gradients to zero
    # because Pytorch accumulation gradients
    w.grad.zero_()
    b.grad.zero_()

preds = model(inputs)
loss = mse(preds, targets)
print(loss)

print("="*40)
# Pytorch built-ins======================
import torch.nn as nn
from torch.utils.data import TensorDataset
# 1. create TensorDataset tuple
train_ds = TensorDataset(inputs, targets)
print(train_ds[0:2])
# 2. create DataLoader, split data into batch
from torch.utils.data import DataLoader
batch_size = 3
train_dl = DataLoader(train_ds,
                      batch_size,
                      shuffle=True)

# instead init weights & biases manually, we define nn.Linear
model = nn.Linear(3, 2)
import torch.nn.functional as F
loss_fn = F.mse_loss
loss = loss_fn(model(inputs), targets)
# define optimizer to update weight and biases
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    for epoch in range(num_epochs):
        for xb, yb in train_dl:
            # 1. generate predictions
            preds = model(xb)
            # 2. Calculate loss
            loss = loss_fn(preds, yb)
            # 3. Compute gradients
            loss.backward()
            # 4. Update parameters using gradients
            opt.step()
            # 5. Reset the gradients to zero
            opt.zero_grad()
        if (epoch+1) % 10 == 0:
            # loss.item() return actual value stored in the loss tensor
            print("Epoch [{}/{}], loss: {:.4f}".format(epoch+1, num_epochs, loss.item()))

# START Training
fit(30, model, loss_fn, opt, train_dl)

