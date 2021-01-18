import torch
import torch.nn as nn
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
# %matplotlib inline indicate config Jupyter plot graph

dataset = MNIST(root="data/", download=True)
image, label = dataset[0]
plt.imshow(image, cmap='gray')
print('Label: ', label)


dataset = MNIST(root="data/",
                train=True,
                transform=transforms.ToTensor())
img_tensor, label = dataset[0]
# image value range from 0 to 1
# - 0 representing black
# - 1 representing white
print(img_tensor[0, 10:15, 10:15])
print(img_tensor.shape, label)

# Training and Validating Datasets
from torch.utils.data import random_split, DataLoader
train_ds, val_ds = random_split(dataset,
                                [50000, 10000])
batch_size = 128
train_loader = DataLoader(train_ds,
                          batch_size,
                          shuffle=True)
val_loader = DataLoader(val_ds,
                        batch_size)
# Model======================================
input_size = 28*28
num_classes = 10
# Logistic regression model
model = nn.Linear(input_size, num_classes)


# evaluate
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# image shape 1x28x28 need convert to vertor of size 784
class MnistModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, xb):
        # view xb with two dim (first dim, 2nd is 784)
        xb = xb.reshape(-1, 784)
        out = self.linear(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images) # generate predict
        loss = F.cross_entropy(out, labels) # cal loss
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        # combine losses
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print('Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}'.format(
            epoch, result['val_loss'], result['val_acc'])
        )

# Note: model no contain .weight, .bias BUT it have .parameters
model = MnistModel()

for images, labels in train_loader:
    print(images.shape)
    outputs = model(images)
    break

# in softmax, replace output y -> e^y (making elements positive)
import torch.nn.functional as F
probs = F.softmax(outputs, dim=1)
print("Sample probabilities:\n", probs[:2].data)
print("Sum: ", torch.sum(probs[0]).item())

# torch.max() return row'largest element and corresponding index
max_probs, preds = torch.max(probs, dim=1)
print("Max probs: ", max_probs)
print("Preds: ", preds)


def evaluation(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    optimizer = opt_func(model.parameters(), lr)
    history = []
    for epoch in range(epochs):
        # training phase
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # validation phase
        result = evaluation(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

history1 = fit(5, 0.001, model, train_loader, val_loader)

accuracies = [result['val_acc'] for result in history1]
plt.plot(accuracies, "-x")
plt.xlabel("epoch")
plt.ylabel("accuracy")
plt.title('Accuracy vs. No. of epochs')


# prediction
def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return preds[0].item()


test_ds = MNIST(root='data/', train=False, transform=transforms.ToTensor())
img, label = test_ds[919]
plt.imshow(img[0], cmap='gray')
print('Label: ', label, ', Predicted:', predict_image(img, model))

# Save and upload
# torch.save(model.state_dict(), 'mnist-logistic.pth')
