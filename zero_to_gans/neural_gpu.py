import torch


def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


# move data and model to device
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """yield a batch of data"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """return number of batch"""
        return len(self.dl)


# yield used to create generator
def some_numbers():
    yield 10
    yield 20
    yield 30


for value in some_numbers():
    print(value)

device = get_default_device()
print(device)