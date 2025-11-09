import torch

x = torch.arange(1, 9)
print(x)
x = x.view(1, 8, 1, 1)

print(x.shape[1])
