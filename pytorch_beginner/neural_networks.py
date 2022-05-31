# Define the network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square conv
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5 * 5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        """
        Define forward function, backward function automatically defined using autograd
        :param x:
        :return:
        """
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # if the size is a square, you can specify with a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = torch.flatten(x, 1)  # flatten all dimensions except the batch dimension
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

params = list(net.parameters())
print(len(params))
print(params[0].size())

# Random 32x32 input and Forward function
input = torch.randn(1, 1, 32, 32)
output = net(input)
target = torch.randn(10)[None]  # a dummy target, for example
criterion = nn.MSELoss()

loss = criterion(output, target)
print(loss)

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# ======= Backpropagate to loss.backward()

net.zero_grad()  # zeroes the gradient buffers of all parameters
print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()
print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# ======== Update the weights simple using Stochastic Gradient Descent: weight = weight - learning_rate * gradient
learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# ======== Various different update rule SGD, Nesterov-SGD, Adam, RMSProp,...
optimizer = optim.SGD(net.parameters(), lr=0.01)
output = net(input)
loss = criterion(output, target)
optimizer.zero_grad()  # because gradient accumulated
loss.backward()
optimizer.step()  # Adjust param by gradients collected in backward pass
