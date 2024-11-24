import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA


device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform  = transforms.ToTensor()
dataset    = datasets.MNIST(root='./MNIST', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

images, _  = next(iter(dataloader))  
images     = images.view(len(dataset), -1).numpy()  

pca = PCA(n_components=400)
pca.fit(images)

vec = pca.transform(images)
print(type(vec))


dist_min = np.inf
vec1_min = 0
vec2_min = 0

for i in range(600):
    for j in range(i+1, 600):
        vec1 = vec[i]
        vec2 = vec[j]

        dist = np.linalg.norm(vec1 - vec2)
        if dist <= dist_min:
            vec1_min = vec1
            vec2_min = vec2
            dist_min = dist

print(dist_min)
print('vec1: ', vec1_min)
print('vec2: ', vec2_min)

reconstracted_image1 = pca.inverse_transform(vec1.reshape(1, -1))
reconstracted_image2 = pca.inverse_transform(vec2.reshape(1, -1))
reconstracted_image1 = reconstracted_image1.reshape(28, 28)
reconstracted_image2 = reconstracted_image2.reshape(28, 28)

fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(reconstracted_image1, cmap='gray')
ax[0].set_title("Image #1")
ax[1].imshow(reconstracted_image2, cmap='gray')
ax[1].set_title("Image #2")
plt.savefig('./sample.png')