import torch
import numpy as np

x = torch.rand(2, 2, dtype=torch.double) #empty(2, 2) zeros(2, 2) ones(2, 2)
y = torch.rand(2, 2, dtype=torch.double)
print(x)
print(x.dtype)
print(x.size())
print(y)

z = x + y
print(z)

z = torch.add(x, y)
print(z)

y.add_(x)
print(y)

z = x - y
z = torch.sub(x, y)
z = torch.mul(x, y)
z = torch.div(x, y)

x = torch.rand(5, 3)
print(x)
print(x[:, 0]) # print the first col
print(x[0, :]) # print the first row
print(x[1, 1].item()) # get the actual value of one item

x = torch.rand(4, 4)
y = x.view(16)
print(x)
print(y)
y = x.view(-1, 8) # -1 determines the right dimension for the tensor

x = torch.tensor([2.5, 1.0])

a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
print(type(b))

a.add_(1)
print(a)
print(b)

a = np.ones(5)
print(a)
b = torch.fron_numpy(a)
print(b)

a += 1
print(a)
print(b)








