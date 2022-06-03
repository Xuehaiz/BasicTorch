import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math

class WineDataset(Dataset):
	def __init__(self):
		xy = np.loadtxt('data/wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
		self.x = torch.from_numpy(xy[:, 1:])
		self.y = torch.from_numpy(xy[:, [0]])
		self.n_samples = xy.shape[0]

	def __getitem__(self, index):
		return self.x[index], self.y[index]

	def __len__(self):
		return self.n_samples

dataset = WineDataset()
# first_data = dataset[0]
# features, labels = first_data
# print(first_data)

dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)
# dataiter = iter(dataloader)
# data = dataiter.next()
# features, labels = data
# print(features, labels)

# training loops 
num_epoches = 2
total_samples = len(dataset)
n_iter = math.ceil(total_samples / 4)
# print(total_samples, n_iter)

for epoch in range(num_epoches):
	for i, (inputs, labels) in enumerate(dataloader):
		# forward, backward and update
		if (i + 1) % 5 == 0:
			print(f'epoch {epoch+1}/{num_epoches}, step {i+1}/{n_iter}, inputs {inputs.shape}')
