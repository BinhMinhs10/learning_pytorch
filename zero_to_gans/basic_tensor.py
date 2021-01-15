import torch
import numpy as np

x = np.array([[1., 2],
              [3, 4],
              [5, 6]])
t1 = torch.from_numpy(x)
# tensor with a fixed value for all element
t2 = torch.full((3, 2), 42)

t3 = torch.cat((t1, t2))
print(t3)

# change the shape of tensor
t4 = t3.reshape(3, 2, 2)
print(t4)
# compute sin of each element
t5 = torch.sin(t4)
