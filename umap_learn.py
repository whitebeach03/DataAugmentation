import umap

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# from sklearn.decomposition import PCA

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform  = transforms.ToTensor()
dataset    = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

images, labels  = next(iter(dataloader))  
images          = images.view(len(dataset), -1).numpy()  

reducer   = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=30)
reducer.fit(images)
embedding = reducer.transform(images)

plt.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap='viridis')
plt.xlabel('UMAP 1st component')
plt.ylabel('UMAP 2nd component')
plt.title('UMAP applied to MNIST')
plt.savefig('umap_mnist.png')